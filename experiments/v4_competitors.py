"""V4 competitors: native training loops, seed-0 selection and matched runs.

Run with python -m experiments.v4_competitors --help. Real sweeps are explicit;
--fake-data is only a two-task, one-epoch CPU fixture and cannot select real configs.
"""
import argparse
import ast
import copy
from contextlib import contextmanager
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from adapters.v4_competitors import CompetitorHook, PINS, ROOT, cflat_module, source_module
from experiments.competitors import cil_runtime
from experiments.runners import pycil as cil, lop
from adapters.cil_setup import class_order, skipinit, FixtureBackbone
from experiments.main_cil import measure_task, training_settings, recipe as main_recipe, NOBN_LR_SCALES
from experiments.vna_provenance import code_commit
from experiments.vna_resume import Progress, Status, rng_state, restore_rng, pause_if_due

ROUND2 = ('podnet', 'wa', 'gem', 'icarl_cbp', 'icarl_redo', 'icarl_snp', 'icarl_upgd', 'icarl_cflat')
ROUND3 = ('icarl_l2init', 'icarl_forget_forgetting', 'icarl_fire', 'icarl_swr', 'icarl_mrfa',
          'bic', 'der', 'foster', 'memo')
SOURCES = {
    'icarl_cbp': dict(source='loss-of-plasticity/lop/algos/res_gnt.py:ResGnT', reimplemented=False),
    'icarl_redo': dict(source='dopamine/dopamine/labs/redo/weight_recyclers.py:BaseRecycler.estimate_neuron_score; NeuronRecycler.recycle_dead_neurons/create_masks; weight_reinit_random/zero; reset_momentum', reimplemented=True),
    'icarl_snp': dict(source='warm_start/run.py:shrink_perturb (126-136), native initialization port', reimplemented=True),
    'icarl_upgd': dict(source='upgd/core/optim/weight_upgd/first_order.py:FirstOrderGlobalUPGD', reimplemented=False),
    'icarl_cflat': dict(source='C-Flat/models/icarl.py:iCaRL; optims/c_flat.py:C_Flat; main.py:setup_parser', reimplemented=False),
    'icarl_fire': dict(source='FIRE/vision/interventions/fire.py:fire, applied when new data arrives as in FIRE/vision/train.py:189-191', reimplemented=False),
    'icarl_swr': dict(source='SWR/src/swr_functions/selective_weight_reinitialization.py:SelectiveWeightReinitialization; config experiments/vit_incremental_cifar/config/incremental_cifar/swr_with_reparam_ln.json; init-parameter table ported to PyCIL resnet32 (adapters/v4_competitors.py:swr_init_parameters)', reimplemented=True),
    'icarl_l2init': dict(source='Kumar et al., Regenerative Regularization (CoLLAs 2024), L2-Init penalty lambda ||theta-theta_init||^2', reimplemented=True),
}
CFLAT_REFERENCE = dict(value=59.13, metric='average accuracy (published head unspecified)', tolerance_pp=1.5,
    source='third_party/C-Flat/resources/cflat_results.png, Table 1, iCaRL w/ C-Flat, B0_Inc5; README.md:20')


def upgd_values():
    values = dict(beta_utility=[], sigma=[])
    for name in ('label_permuted_emnist.py', 'label_permuted_cifar10.py'):
        tree = ast.parse((ROOT/'third_party/upgd/experiments'/name).read_text())
        call = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'up_grids' for t in node.targets))
        for kw in call.keywords:
            if kw.arg in values: values[kw.arg].extend(ast.literal_eval(kw.value))
    return {k: sorted(set(v)) for k, v in values.items()}


def grid(method):
    values = {
        'icarl_cbp': dict(replacement_rate=[1e-6,1e-5,1e-4,1e-3]),
        'icarl_redo': dict(tau=[0.,.025,.1], reset_epochs=[1,5]),
        'icarl_snp': dict(shrink=[.4,.8], sigma=[1e-4,1e-2]),
        'icarl_fire': dict(iter_num=[10]),
        'icarl_swr': dict(utility_function=['gradient'], pruning_method=['threshold'],
                          reinit_freq=[128], reinit_factor=[5e-4]),
    }.get(method, upgd_values() if method == 'icarl_upgd' else {})
    return [dict(zip(values, v)) for v in itertools.product(*values.values())] if values else [{}]


def recipe(method, backbone, seed, device, reproduction=False, nobn_lr_scale=1.):
    source = method if method in ('podnet','wa','gem','bic','der','foster','memo') else 'icarl'
    path = ROOT/('third_party/C-Flat/exps/icarl.json' if reproduction or method == 'icarl_cflat'
                 else f'third_party/PyCIL/exps/{source}.json')
    original = json.loads(path.read_text())
    cfg = copy.deepcopy(original)
    common = main_recipe(seed, device, 'resnet32')
    cfg.update({k: common[k] for k in ('init_cls','increment','memory_size','fixed_memory','dataset','seed','device')})
    cfg.update(init_cls=5, increment=5, memory_size=2000, fixed_memory=False,
               dataset='cifar100', shuffle=reproduction, seed=[seed],
               device=[-1] if device == 'cpu' else [device.split(':')[-1]])
    if source not in ('podnet','memo'): cfg['convnet_type'] = 'resnet32'
    if method == 'icarl_cflat': cfg.update(cflat=True, rho=.2, lamb=.2)
    return cfg, str(path.relative_to(ROOT)), {k: dict(upstream=original.get(k), value=v)
                                            for k,v in cfg.items() if original.get(k) != v}


def prepare_method(method):
    if method == 'gem':
        try: import quadprog  # noqa: F401
        except ImportError as error:
            raise RuntimeError('GEM requires quadprog; install requirements-upstream.txt first') from error
    if method == 'icarl_mrfa':
        raise RuntimeError('MRFA (third_party/MRFA, 2435c35) is a PyCIL fork with its own models/ and utils/ packages; it needs a dedicated runner that does not share the PyCIL import namespace')
    if method == 'icarl_forget_forgetting':
        raise RuntimeError('Forget-Forgetting needs the paper/supplementary reset and consolidation equations; no official code or full paper is present locally')


@contextmanager
def runtime(args, method, cfg, boundary, epoch_done):
    source = cfg['model_name']
    with cil_runtime(source, fake=args.fake_data, data_dir=args.data_dir, boundary=boundary) as native:
        module = cflat_module() if method == 'icarl_cflat' else native
        from utils import inc_net, data_manager
        original_dtype = torch.get_default_dtype()
        if args.fake_data: torch.set_default_dtype(torch.float64)
        original_get = inc_net.get_convnet
        original_data = data_manager._get_idata
        original_loader = module.DataLoader
        saved = {k: copy.deepcopy(getattr(module, k)) for k in
                 ('init_epoch','epochs','ft_epochs','init_milestones','milestones','init_lr','lrate') if hasattr(module,k)}
        order = class_order(args.seed, args.order, None if args.fake_data else args.data_dir)
        def data(name):
            result = original_data(name)
            result.class_order = order
            if args.fake_data:
                from torchvision import transforms
                result.common_trsf = [*result.common_trsf, transforms.ConvertImageDtype(torch.float64)]
            return result
        def backbone(params, pretrained=False):
            if args.fake_data and params['convnet_type'] in ('resnet32', 'cosine_resnet32'):
                name = 'ucir_cifar_resnet' if params['convnet_type'].startswith('cosine') else 'cifar_resnet'
                conv = importlib.import_module('convs.'+name)
                net = FixtureBackbone(conv.ResNetBasicblock).double()
            else: net = original_get(params, pretrained)
            if isinstance(net, tuple):
                if args.backbone == 'nobn':
                    for part in net: skipinit(part)
            else:
                # This unused native head has no gradients; GEM/UPGD require active parameters.
                if hasattr(net, 'fc'): del net.fc
                if args.backbone == 'nobn': skipinit(net)
            return net
        class Loader(original_loader):
            def __init__(self, *a, **kw):
                if args.fake_data: kw['num_workers'] = 0
                super().__init__(*a, **kw)
            def __iter__(self):
                for batch in super().__iter__():
                    boundary()
                    yield batch
                if getattr(self, '_v4_training', False): epoch_done()
        inc_net.get_convnet = backbone
        data_manager._get_idata = data
        module.DataLoader = Loader
        try:
            yield module, order
        finally:
            torch.set_default_dtype(original_dtype)
            inc_net.get_convnet = original_get
            data_manager._get_idata = original_data
            module.DataLoader = original_loader
            for k,v in saved.items(): setattr(module,k,v)


def set_epochs(module, learner, args, method):
    first = learner._cur_task < 0
    scale = args.nobn_lr_scale if args.backbone == 'nobn' else 1.
    settings = training_settings(args.fake_data, args.reproduction, scale)
    total = settings['init_epoch' if first else 'epochs']
    for name, value in settings.items():
        if hasattr(module,name): setattr(module,name,value)
    if hasattr(module,'ft_epochs'):
        module.ft_epochs = 0 if args.fake_data else 20
        module.epochs = total if first else total-module.ft_epochs
    cfg = learner.args
    original = json.loads((ROOT/('third_party/C-Flat/exps/icarl.json' if method == 'icarl_cflat'
                                else 'third_party/PyCIL/exps/'+cfg['model_name']+'.json')).read_text())
    for name, value in settings.items():
        if name in cfg: cfg[name] = value
    if 'init_epochs' in cfg: cfg['init_epochs'] = settings['init_epoch']
    if 'boosting_epochs' in cfg:
        boost = round(total*original['boosting_epochs']/(original['boosting_epochs']+original['compression_epochs']))
        cfg['boosting_epochs'],cfg['compression_epochs'] = max(1,boost),max(1,total-boost)


def run(args):
    if args.backbone=='nobn' and args.nobn_lr_scale!=.5:raise ValueError('noBN requires nobn_lr_scale=0.5')
    method = 'icarl' if args.disabled else args.method
    prepare_method(method)
    if args.reproduction and (args.method != 'icarl_cflat' or args.backbone != 'bn' or args.order != 'random' or args.seed != 1993 or args.disabled):
        raise ValueError('C-Flat reproduction requires icarl_cflat, BN, random order, seed 1993, enabled')
    if not args.fake_data and not args.data_dir: raise ValueError('--data-dir is required')
    config = selected_config(args) if args.best_config else (json.loads(args.config) if args.config else grid(method)[0])
    if method == 'icarl': config = {}
    if method == 'icarl_l2init':
        if 'coefficient' not in config: raise ValueError('L2-Init requires --config with coefficient')
    elif config not in grid(method): raise ValueError('configuration outside registered grid')
    cil.upstream_imports()
    cfg, source, overrides = recipe(method,args.backbone,args.seed,args.device,args.reproduction,args.nobn_lr_scale)
    if args.fake_data:
        torch.set_num_threads(1)
        cfg.update(memory_size=20, memory_per_class=2)
    from experiments.parallel_validity import dataset_fingerprint
    identity = dict(benchmark='v4_competitor', arm=args.method, effective_method=method, seed=args.seed,
        backbone=args.backbone, order=args.order, nobn_lr_scale=args.nobn_lr_scale, cfg_source=source,
        config_overrides=overrides, cfg=copy.deepcopy(cfg), method_config=config,
        fake_data=args.fake_data, disabled=args.disabled, reproduction=args.reproduction,
        code_commit=code_commit(), source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
        upstream_commit=cil.UPSTREAM_PIN, competitor_pins=PINS,
        implementation=SOURCES.get(method, dict(source=f'PyCIL/models/{method}.py', reimplemented=False)),
        expected_tasks=2 if args.fake_data else 20,
        class_order=class_order(args.seed,args.order,None if args.fake_data else args.data_dir),
        training=training_settings(args.fake_data,args.reproduction,args.nobn_lr_scale if args.backbone=='nobn' else 1.),
        probe=dict(size=4 if args.fake_data else 256,seed=0,source='seen test classes'),
        feature_probe=dict(tasks=[5,10,15,20],optimizer='LBFGS',max_iter=100,initialization='zeros',
                           data='all seen train, test transforms',regularization=0),
        dataset_fingerprint=None if args.fake_data else dataset_fingerprint(args.data_dir),
        initialization_seed=1 if args.reproduction else args.seed,
        epochs='1/1 fixture' if args.fake_data else ('200/170' if args.reproduction else '100/50'),
        fixture='two-channel float64 residual backbone, 8 images/class, memory 20' if args.fake_data else None,
        resume='exact task-boundary restart, including method state and RNG')
    out = Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    progress = Progress(out, identity)
    if progress.state and not args.resume: raise ValueError('existing checkpoint requires --resume')
    if args.resume and (out/'summary.json').exists():
        previous = json.loads((out/'summary.json').read_text())
        if previous.get('identity') != identity: raise ValueError('completed identity mismatch')
        if previous.get('status') == 'COMPLETED': return previous
    for name, pin in PINS.items():
        actual = subprocess.check_output(['git','-C',str(ROOT/'third_party'/name),'rev-parse','HEAD'],text=True).strip()
        if actual != pin: raise ValueError(f'{name} pin mismatch')
    lop.write_json(out/'protocol.json',identity)
    status = Status(out,args.method)
    tasks, diagnostics = [], []
    started, base_seconds = time.monotonic(), 0.
    learner, hook = None, None
    def boundary():
        def paused():
            if progress.path.exists():
                saved = torch.load(progress.path,map_location='cpu',weights_only=False)
                saved['elapsed_seconds'] = base_seconds+time.monotonic()-started
                progress.save(saved)
            lop.write_json(out/'summary.json',dict(status='PAUSED',identity=identity,resume_supported=True))
            status.update(phase='PAUSED')
        pause_if_due(paused)
    def epoch_done():
        if hook: hook.after_epoch(learner)
    try:
        with lop.working_directory(out), runtime(args,method,cfg,boundary,epoch_done) as (module,order):
            import trainer
            from utils.data_manager import DataManager
            from utils import factory
            random.seed(args.seed); np.random.seed(args.seed)
            if args.reproduction:
                source_module('v4_cflat_trainer','C-Flat/trainer.py')._set_random()
            else:
                trainer._set_random()
                torch.manual_seed(args.seed)
            trainer._set_device(cfg)
            manager = DataManager('cifar100',args.reproduction,args.seed,5,5)
            if args.fake_data: manager._increments = manager._increments[:2]
            lop.write_json(out/'class_order.json',order)
            learner = module.iCaRL(cfg) if method == 'icarl_cflat' else factory.get_model(cfg['model_name'],cfg)
            if args.fake_data: learner._network.double()
            if progress.state:
                learner = progress.state['model']
                tasks, diagnostics = progress.state['tasks'], progress.state['diagnostics']
                base_seconds = progress.state['elapsed_seconds']
                for value in vars(learner).values():
                    if isinstance(value, torch.nn.Module): value.to(cfg['device'][0])
                restore_rng(progress.state['rng'])
            def checkpoint():
                state = copy.deepcopy(learner)
                for key in ('train_loader','test_loader','val_loader'): vars(state).pop(key,None)
                progress.save(dict(model=state,tasks=tasks,diagnostics=diagnostics,rng=rng_state(),
                    elapsed_seconds=base_seconds+time.monotonic()-started))
                lop.write_json(out/'tasks.json',tasks)
                lop.write_json(out/'diagnostics.json',diagnostics)
                lop.write_csv(out/'continual_eval.csv',tasks)
            if not progress.state: checkpoint()
            for task in range(len(tasks),manager.nb_tasks):
                boundary(); status.update(task=task+1,phase='native training')
                set_epochs(module,learner,args,method)
                original_train = type(learner)._train
                original_run = getattr(type(learner),'_run',None)
                def train(model,*a,**kw):
                    for loader in a:
                        if isinstance(loader,torch.utils.data.DataLoader) and loader is model.train_loader:
                            loader._v4_training = True
                    if hook:
                        with hook.install(model,module): return original_train(model,*a,**kw)
                    return original_train(model,*a,**kw)
                if original_run:
                    def native_run(model,*a,**kw):
                        if a[-1] == 0: return
                        return original_run(model,*a,**kw)
                    type(learner)._run = native_run
                hook = CompetitorHook(method,config) if method in ('icarl_cbp','icarl_redo','icarl_snp','icarl_upgd','icarl_fire','icarl_swr') else None
                handles = []
                if method == 'icarl_l2init':
                    initial = getattr(learner,'_v4_initial',{})
                    # Head grows: keep the original initialization for existing rows.
                    def l2_train(model,*a,**kw):
                        for n,p in model._network.named_parameters():
                            target = p.detach().clone()
                            if n in initial:
                                old = initial[n].to(p)
                                if old.shape == p.shape: target = old
                                elif old.ndim == p.ndim: target[:len(old)] = old
                            initial[n] = target
                            handles.append(p.register_hook(lambda grad,p=p,target=target: grad+2*config['coefficient']*(p-target)))
                        return original_train(model,*a,**kw)
                    learner._v4_initial = initial
                    train = l2_train
                type(learner)._train = train
                try: learner.incremental_train(manager)
                finally:
                    type(learner)._train = original_train
                    if original_run: type(learner)._run = original_run
                    for handle in handles: handle.remove()
                if not all(torch.isfinite(p).all() for p in learner._network.parameters()):
                    status.update(phase='DIVERGED',nonfinite_params=True)
                    result = dict(status='DIVERGED',identity=identity,task=task+1,nonfinite_params=True,tasks=tasks)
                    lop.write_json(out/'summary.json',result)
                    return result
                row, diagnostic = measure_task(learner,manager,tasks,boundary,4 if args.fake_data else 256)
                diagnostics.append(diagnostic)
                learner.after_task()
                tasks.append(row); checkpoint()
            seconds = base_seconds+time.monotonic()-started
            result = dict(status='COMPLETED',identity=identity,tasks=tasks,tasks_run=len(tasks),
                arm=args.method,seed=args.seed,init_cls=5,increment=5,parameter_count=tasks[-1]['parameter_count'],
                model_group='expanding' if method in ('der','foster','memo') else 'fixed',
                seconds=seconds,gpu_hours=seconds/3600 if args.device != 'cpu' else 0.,
                average_incremental_accuracy=tasks[-1]['cnn_average_accuracy'],
                last_accuracy=tasks[-1]['cnn_top1'],resume_supported=True)
            for label in ('cnn','nme'):
                result.update({label+'_'+name: tasks[-1][label+'_'+name] for name in
                               ('average_accuracy','last_accuracy','forgetting')})
                result[label+'_learning_accuracy'] = float(np.mean([t[label+'_learning_accuracy'] for t in tasks]))
            if args.reproduction:
                difference = result['average_incremental_accuracy']-CFLAT_REFERENCE['value']
                result['reproduction'] = dict(**CFLAT_REFERENCE,delta_cnn_pp=difference,
                    delta_nme_pp=tasks[-1].get('nme_average_accuracy',float('nan'))-CFLAT_REFERENCE['value'],
                    status='INVALID' if args.fake_data else 'UNVERIFIED_HEAD')
                lop.write_json(out/'reproduction_gate.json',result['reproduction'])
            lop.write_json(out/'summary.json',result)
            status.update(phase='COMPLETED')
            return result
    except (AssertionError,FloatingPointError,torch.linalg.LinAlgError) as error:
        diverged = not isinstance(error,AssertionError)
        result = dict(status='DIVERGED' if diverged else 'INVALID',identity=identity,tasks=tasks,
                      task=len(tasks)+1,step=hook.steps if hook else None,
                      nonfinite_params=bool(learner is not None and any(not torch.isfinite(p).all() for p in learner._network.parameters())),
                      error=f'{type(error).__name__}: {error}')
        lop.write_json(out/'summary.json',result)
        status.update(phase=result['status'],nonfinite_params=result['nonfinite_params'],step=result['step'])
        return result
    finally: status.close()


def reuse_gem_candidate(args):
    if args.method!='gem' or args.fake_data or not args.reuse_root:return None
    from scripts.collect_parallel import validate_result
    from experiments.parallel_validity import dataset_fingerprint
    import hashlib
    import shutil
    expected=dataset_fingerprint(args.data_dir);matches={}
    target=Path(args.out).resolve()
    for path in Path(args.reuse_root).rglob('summary.json'):
        if path.parent.resolve()==target and not (target/'reused_candidate.json').exists():continue
        value=json.loads(path.read_text());p=value.get('identity',{})
        if value.get('status')!='COMPLETED' or any(p.get(k)!=v for k,v in dict(
            benchmark='v4_competitor',arm='gem',seed=0,backbone='nobn',order='superclass',
            nobn_lr_scale=args.nobn_lr_scale,method_config=json.loads(args.config),dataset_fingerprint=expected).items()):continue
        validate_result('gem__nobn__superclass__s0',value,p,value.get('tasks'))
        if p.get('class_order')!=class_order(0,'superclass',args.data_dir) or p.get('training')!=training_settings(False,False,args.nobn_lr_scale):
            raise ValueError('GEM reuse order/training mismatch')
        matches[hashlib.sha256(path.read_bytes()).hexdigest()]=(path,value)
    if len(matches)>1:raise ValueError('Conflicting existing GEM seed0 candidates')
    if not matches:return None
    digest,(path,value)=next(iter(matches.items()))
    names=('summary.json','protocol.json','tasks.json','diagnostics.json','continual_eval.csv','status.json')
    if any(not path.with_name(name).exists() for name in names):raise ValueError('GEM reuse artifact missing')
    protocol=json.loads(path.with_name('protocol.json').read_text())
    tasks=json.loads(path.with_name('tasks.json').read_text())
    validate_result('gem__nobn__superclass__s0',value,protocol,tasks)
    import csv
    if len(json.loads(path.with_name('diagnostics.json').read_text()))!=20 or json.loads(path.with_name('status.json').read_text()).get('phase')!='COMPLETED':
        raise ValueError('GEM reuse diagnostics/status incomplete')
    if len(list(csv.DictReader(path.with_name('continual_eval.csv').read_text().splitlines())))!=20:
        raise ValueError('GEM reuse evaluation incomplete')
    if path.parent.resolve()==target:
        metadata=json.loads((target/'reused_candidate.json').read_text())
        if metadata.get('consumer_commit')!=code_commit() or metadata.get('summary_sha256')!=digest:
            raise ValueError('GEM reuse cache identity mismatch')
        return value
    target.mkdir(parents=True,exist_ok=True)
    for name in names:shutil.copy2(path.with_name(name),target/name)
    print(f'GEM REUSE: validated seed0 lr={.1*args.nobn_lr_scale:g} from {path}',flush=True)
    lop.write_json(target/'reused_candidate.json',dict(source=str(path),summary_sha256=digest,
        source_commit=value['identity']['code_commit'],consumer_commit=code_commit(),scope='GEM seed0 at this exact LR/config'))
    return value


def sweep(args):
    if args.disabled or args.reproduction: raise ValueError('sweep requires an enabled main-protocol method')
    out = Path(args.out)
    candidates = []
    trials=[]
    for j,scale in enumerate(NOBN_LR_SCALES):
        for i,config in enumerate(grid(args.method)):
            trial = copy.deepcopy(args)
            trial.seed, trial.backbone, trial.order = 0,'nobn','superclass'
            trial.nobn_lr_scale=scale;trial.best_config=None
            relative=f'scan/lr{j}/{i}'
            trial.config, trial.out = json.dumps(config), str(out/relative)
            print(f'SWEEP {args.method}: lr={.1*scale:g} config={config} out={trial.out}',flush=True)
            result = reuse_gem_candidate(trial) or run(trial)
            score=result.get('average_incremental_accuracy',float('nan'))
            trials.append(dict(config=config,nobn_lr_scale=scale,lr=.1*scale,path=relative,
                status=result['status'],score=score if math.isfinite(score) else None))
            lop.write_json(out/'selection_report.json',dict(lr_scales=list(NOBN_LR_SCALES),trials=trials))
            if result['status']=='COMPLETED' and math.isfinite(score):candidates.append((config,scale,relative,result))
    if not candidates: raise ValueError('no completed finite sweep candidates')
    config,scale,relative,result=max(candidates,key=lambda item:item[3]['average_incremental_accuracy'])
    best = dict(method=args.method,config=config,seed=0,backbone='nobn',order='superclass',
        metric='cnn_average_accuracy',score=result['average_incremental_accuracy'],
        synthetic=args.fake_data,code_commit=code_commit(),source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
        dataset_fingerprint=result.get('identity',{}).get('dataset_fingerprint'),
        nobn_lr_scale=scale,lr=.1*scale,lr_scales=list(NOBN_LR_SCALES),grid=grid(args.method),
        selected_trial=relative,total_trials=len(trials),
        selection='maximum finite average incremental accuracy at seed0; first registered LR/config wins ties')
    lop.write_json(out/'best_config.json',best)
    return best


def selected_config(args):
    if args.best_config:
        best = json.loads(Path(args.best_config).read_text())
        if best.get('synthetic') != args.fake_data or best.get('method') != args.method or best.get('code_commit') != code_commit() or best.get('source_digest') != os.environ.get('VNA_SOURCE_DIGEST'):
            raise ValueError('best_config provenance mismatch')
        if best.get('grid') != grid(args.method) or best.get('config') not in grid(args.method) or best.get('lr_scales')!=list(NOBN_LR_SCALES):
            raise ValueError('best_config grid mismatch')
        if (best.get('seed'),best.get('backbone'),best.get('order'),best.get('metric')) != (0,'nobn','superclass','cnn_average_accuracy') or best.get('nobn_lr_scale') not in NOBN_LR_SCALES or not math.isfinite(best.get('score',float('nan'))):
            raise ValueError('best_config selection policy mismatch')
        if not args.fake_data:
            from experiments.parallel_validity import dataset_fingerprint
            if best.get('dataset_fingerprint') != dataset_fingerprint(args.data_dir):
                raise ValueError('best_config dataset mismatch')
        args.nobn_lr_scale=best['nobn_lr_scale']
        config = best['config']
    elif len(grid(args.method)) == 1: config = grid(args.method)[0]
    else: raise ValueError('--best-config is required for tuned methods')
    return config


def matched(args):
    config = selected_config(args)
    results = []
    for backbone in ('nobn','bn'):
        for seed in range(args.seeds):
            trial = copy.deepcopy(args)
            trial.backbone,trial.seed,trial.order = backbone,seed,'superclass'
            trial.config,trial.out = json.dumps(config),str(Path(args.out)/backbone/f'seed{seed}')
            results.append(run(trial))
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=('icarl',)+ROUND2+ROUND3,required=True)
    p.add_argument('--mode',choices=('run','sweep','matched'),default='run')
    p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--seeds',type=int,choices=(3,5),default=3)
    p.add_argument('--backbone',choices=('bn','nobn'),default='nobn')
    p.add_argument('--order',choices=('superclass','random'),default='superclass')
    p.add_argument('--nobn-lr-scale',type=float,choices=NOBN_LR_SCALES,default=.5)
    p.add_argument('--device',default='cpu')
    p.add_argument('--data-dir',default='/kaggle/input/datasets/nlnk1607/cifar100')
    p.add_argument('--config'); p.add_argument('--best-config')
    p.add_argument('--reuse-root',help='Expanded prior output tree for validated GEM seed0 sweep-candidate reuse')
    p.add_argument('--fake-data',action='store_true'); p.add_argument('--disabled',action='store_true')
    p.add_argument('--resume',action='store_true'); p.add_argument('--reproduction',action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    {'run':run,'sweep':sweep,'matched':matched}[args.mode](args)
