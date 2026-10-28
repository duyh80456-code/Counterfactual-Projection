"""H3 fixed-capacity tangent reconfiguration on the pinned LoP regression stream."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import random
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from experiments.runners import lop
from experiments.standard_cl import ExperimentLogger
from experiments.vna_provenance import code_commit
from experiments.vna_resume import Progress,rng_state,restore_rng,deadline_due,log
from methods.cl_h3 import TangentReconfiguration
from methods.cl_replay import CorrectionConfig,LossComponent,ReplayCorrection
from methods.vna_losses import MSE
from methods.vna_sites import HookSite


class RegressionNetwork(nn.Module):
    def __init__(self,native):super().__init__();self.native=native
    @property
    def layers(self):return self.native.layers
    @property
    def act_type(self):return self.native.act_type
    def predict(self,x):return self.native.predict(x)
    def forward(self,x):return self.predict(x)[0]
    def sites(self):return [HookSite(self,'native.layers.0','native.layers.2','linear',between=('native.layers.1',))]


def algorithm_state(algorithm):
    return deepcopy({k:v for k,v in vars(algorithm).items() if k not in ('net','opt','loss_func','m','gen_new_features')})


def run(args):
    lop.upstream_imports()
    from lop.nets.ffnn import FFNN
    from lop.algos.bp import Backprop
    from lop.algos.cbp import ContinualBackprop
    from lop.algos.gntRedo import GnTredo
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    logger=ExperimentLogger(args.out);out=Path(args.out).resolve()
    cfg=json.loads(Path(args.config).read_text())
    if min(cfg['steps'],cfg['task_steps'],cfg['window'],cfg['intervention_every'])<1:raise ValueError('positive stream/cadence/window required')
    if not args.fake_data and (cfg['steps']%10000 or cfg['task_steps']%10000):
        raise ValueError('the unmodified upstream generator requires full 10000-point chunks')
    params,source=lop.recipe('regression','cbp' if args.arm=='cbp' else 'bp',cfg['replacement_rate'] if args.arm=='cbp' else None)
    params.update(num_data_points=cfg['steps'],flip_after=cfg['task_steps'])
    if args.fake_data:params.update(num_data_points=32,flip_after=16)
    if params['num_data_points']%params['flip_after'] or params['flip_after']<2:raise ValueError('complete tasks required')
    if args.arm=='snp':
        from experiments.competitors import b5_recipe
        native,source_path,_=b5_recipe('snp',cfg['perturb_scale'])
        params.update(weight_decay=native['weight_decay'],perturb_scale=native['perturb_scale'])
        source=dict(source,cfg_source=source_path)
    from diagnostics.vna_align import _global_state
    # Existing/generated input files must not change the learner's initialization RNG.
    with _global_state():lop.prepare_data('regression',params,out,fake=args.fake_data)
    with open(params['env_file'],'rb') as stream:inputs,targets,_=pickle.load(stream)
    inputs,targets=inputs[:int(params['num_data_points'])],targets[:int(params['num_data_points'])]
    model=RegressionNetwork(FFNN(params['num_inputs'],params['num_features'],hidden_activation=params['hidden_activation']))
    learner=ContinualBackprop(model,step_size=params['step_size'],replacement_rate=params['replacement_rate'],
        decay_rate=params['decay_rate'],maturity_threshold=params['gir_maturity'],util_type=params.get('util_type','adaptable_contribution'),
        accumulate=params.get('accumulate',False)) if args.arm=='cbp' else Backprop(model,step_size=params['step_size'],
        weight_decay=params.get('weight_decay',0.),to_perturb=args.arm=='snp',perturb_scale=params.get('perturb_scale',0.))
    recycler=GnTredo(model.layers,model.act_type) if args.arm=='redo' else getattr(learner,'gnt',None)
    replaced=0
    if recycler is not None:
        original=recycler.gen_new_features
        def counted(features_to_replace,num_features_to_replace):
            nonlocal replaced
            replaced+=sum(int(n) for n in num_features_to_replace)
            return original(features_to_replace,num_features_to_replace)
        recycler.gen_new_features=counted
    numerical=dict(cfg['correction'])
    if args.fake_data:numerical.update(damping_relative=1.,cg_tolerance=1e-4)
    config=CorrectionConfig(**numerical,seed=args.seed)
    engine=TangentReconfiguration(config)
    corrector=ReplayCorrection(config) if args.arm=='vna_gir' else None
    protocol=dict(branch='H3',code_commit=code_commit(),upstream_commit=lop.UPSTREAM_PIN,arm=args.arm,seed=args.seed,
        config=cfg,tasks=len(inputs)//params['flip_after'],params={k:v for k,v in params.items() if k not in ('env_file','data_file')},
        correction=asdict(config),upstream_recipe=source,synthetic_smoke=args.fake_data,
        data_sha256=hashlib.sha256(inputs.contiguous().numpy().tobytes()+targets.contiguous().numpy().tobytes()).hexdigest(),
        data_generation='synthetic binary/sum fixture' if args.fake_data else 'pinned LoP slowly_changing_regression.generate_problem_data; isolated seeded RNG',
        baseline_implementation='pinned upstream Backprop/ContinualBackprop/GnTredo; supervised ReDo hook after native SGD',
        normalization='none; upstream ReLU B5',device='cpu',evidence_status='INCONCLUSIVE')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('plasticity protocol mismatch')
    logger.save('protocol',protocol)
    progress=Progress(out,protocol);position=0;rows=[];curves=[];elapsed_base=0.;last_event=0;gradient_sum=0.;task_replacements=0
    if progress.state:
        s=progress.state;progress.restore_artifacts();model.load_state_dict(s['model']);learner.opt.load_state_dict(s['optimizer'])
        if recycler is not None:vars(recycler).update(s['recycler'])
        engine.load_state_dict(s['engine']);restore_rng(s['rng']);position=s['position'];rows=s['rows'];curves=s['curves']
        if corrector is not None:corrector.load_state_dict(s['vna_engine'])
        elapsed_base=s['elapsed_seconds'];last_event=s['last_event'];replaced=s['replaced']
        gradient_sum=s['gradient_sum'];task_replacements=s['task_replacements']
    started=time.monotonic();status='COMPLETED';failure=None;coordinates=None
    def save():
        logger.save('tasks',rows);logger.save('interventions',engine.events);logger.save('learning_curves',curves)
        if corrector is not None:logger.save('vna_interventions',corrector.events)
        progress.save(dict(model=model.state_dict(),optimizer=learner.opt.state_dict(),recycler=None if recycler is None else algorithm_state(recycler),
            engine=engine.state_dict(),vna_engine=None if corrector is None else corrector.state_dict(),rng=rng_state(),position=position,rows=rows,curves=curves,last_event=last_event,replaced=replaced,
            gradient_sum=gradient_sum,task_replacements=task_replacements,elapsed_seconds=elapsed_base+time.monotonic()-started))
    gradient_values=[]
    handles=[p.register_hook(lambda g:gradient_values.append(float(g.double().square().sum()))) for p in model.parameters()]
    try:
        while position<len(inputs):
            if deadline_due():save();raise SystemExit(75)
            cadence=16 if args.fake_data else cfg['intervention_every']
            pending=args.arm in ('h3','vna_gir') and position+1-last_event>=cadence
            if pending:
                from methods.cl_state import State
                snapshot=State(model,learner.opt);old_gradient_sum=gradient_sum
                if corrector is not None:old_corrector=corrector.state_dict()
            x,y=inputs[position:position+1],targets[position:position+1]
            gradient_values.clear();err=learner.learn(x,y);gradient_sum+=sum(gradient_values)
            if not bool(torch.isfinite(err)):
                coordinates=dict(task=position//params['flip_after']+1,step=position+1,tensor='loss')
                raise FloatingPointError(str(coordinates))
            if args.arm=='redo':recycler.gen_and_test(torch.stack(learner.previous_features,dim=1))
            position+=1;curves.append(float(err))
            if args.arm in ('h3','vna_gir') and position-last_event>=cadence:
                start=max(0,position-cfg['window'])
                target=targets[start:position]
                losses=SimpleNamespace(targets=target,spec=MSE(reduction='mean'),ctx=dict(targets=target),
                    components=(LossComponent('regression_mse',lambda z:(z-target).square().mean()),LossComponent('reference_zero',lambda z:z.sum()*0)))
                try:
                    vna=None
                    if corrector is not None:
                        vna=corrector.step(model,learner.opt,inputs[start:position],losses,model.sites(),params['step_size'])
                        vna.update(task=(position-1)//params['flip_after']+1,step=position)
                    event=engine.step(model,learner.opt,inputs[start:position],losses,model.sites())
                    if vna is not None:event['vna']=vna
                except SystemExit as exc:
                    if exc.code!=75:raise
                    if corrector is not None:corrector.load_state_dict(old_corrector)
                    snapshot.restore();gradient_sum=old_gradient_sum;position-=1;curves.pop();save();raise
                last_event=position
                event.update(task=(position-1)//params['flip_after']+1,step=position)
            if position%params['flip_after']==0:
                errors=np.array(curves[-params['flip_after']:]);quarter=max(1,len(errors)//4)
                h3_replacements=sum(e.get('actual_replacements',0) for e in engine.events)
                rows.append(dict(task=position//params['flip_after'],mean_error=float(errors.mean()),
                    early_error=float(errors[:quarter].mean()),late_error=float(errors[-quarter:].mean()),
                    learning_speed=float(errors[:quarter].mean()-errors[-quarter:].mean()),
                    learning_slope=float(np.polyfit(np.arange(len(errors)),errors,1)[0]),
                    gradient_rms=float((gradient_sum/params['flip_after'])**.5),
                    replacements=replaced+h3_replacements-task_replacements))
                task_replacements=replaced+h3_replacements;gradient_sum=0.;save();log(f'H3 {args.arm} task={rows[-1]["task"]} mse={rows[-1]["mean_error"]:.6f}')
            elif deadline_due():save();raise SystemExit(75)
    except SystemExit as exc:status='PAUSED' if exc.code==75 else 'INTERRUPTED';raise
    except FloatingPointError as exc:status='FAILED_DIVERGED';failure=str(exc)
    except BaseException as exc:status='FAILED_ERROR';failure=repr(exc);raise
    finally:
        for handle in handles:handle.remove()
        seconds=elapsed_base+time.monotonic()-started
        logger.save('summary',dict(protocol=protocol,status=status,failure=failure,diverged=coordinates,tasks=rows,
            last_50_task_error=float(np.mean([r['mean_error'] for r in rows[-50:]])) if rows else None,
            plasticity_decay=rows[-1]['late_error']-rows[0]['late_error'] if rows else None,
            wall_seconds=seconds,correction_seconds=sum(e['seconds'] for e in engine.events)+(sum(e['seconds'] for e in corrector.events) if corrector is not None else 0.),gpu_hours=0.,peak_vram_bytes=0,
            parameter_count=sum(p.numel() for p in model.parameters()),evidence_status='INCONCLUSIVE'))
    return json.loads((out/'summary.json').read_text())


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/standard_cl/h3_regression.json');p.add_argument('--out',required=True)
    p.add_argument('--arm',choices=('bp','cbp','redo','snp','h3','vna_gir'),default='h3');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--fake-data',action='store_true',help='32-point synthetic control; no plasticity conclusion')
    return p


if __name__=='__main__':run(parser().parse_args())
