"""P1 full-chain baseline/full comparison with isolated trajectory measurements."""
import copy
import hashlib
from contextlib import ExitStack
import json
import math
import os
import re
from pathlib import Path
import time
import numpy as np
import torch
from diagnostics.stability_gap import Tracker, read, write
from diagnostics.vna_align import _global_state, _assert_bitwise
from experiments import main_cil as main
from experiments.vna_resume import rng_state
from methods.vna_v3 import V3Engine

POINTS = (0,1,2,5,10,20,50,100,200)


def validate_nb1_baseline(path, seed, data_dir):
    return validate_p1_baseline(path, seed, data_dir, 20)


def validate_p1_baseline(path, seed, data_dir, increment):
    """Reuse registered pre-amendment results without relabeling their source."""
    from experiments.parallel_validity import dataset_fingerprint
    if (increment,seed) not in ((20,0),(20,1),(5,2)):
        raise ValueError('unregistered baseline reuse configuration')
    label = 'NB1' if increment==20 else 'M1 seed2'
    path = Path(path)
    summary_path = path/'summary.json'
    if not summary_path.is_file(): summary_path = path/'run'/'summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError(f'{label} baseline summary not restored: {path}/summary.json or {summary_path}; attach original baseline Output')
    summary = read(summary_path, {})
    p = summary.get('identity', {})
    outcome = summary.get('status')
    failed_reference = increment==5 and seed==2 and outcome=='FAILED_DIVERGED'
    if outcome != 'COMPLETED' and not failed_reference:
        raise ValueError(f'{label} baseline not completed: path={path}, status={summary.get("status")!r}, tasks={len(summary.get("tasks",[]))}')
    divergence = summary.get('diverged') if failed_reference else None
    if failed_reference and (not isinstance(divergence,dict)
            or any(type(divergence.get(k)) is not int or divergence[k]!=v
                   for k,v in (('task',2),('epoch',1),('step',8)))
            or not isinstance(divergence.get('tensor'),str) or not divergence['tensor'].strip()):
        raise ValueError(label+' baseline missing or mismatched divergence coordinates (task=2, epoch=1, step=8, tensor required)')
    sources = {('ccf60af0909afa8853357c660f8760c98cc1b828', digest) for digest in (
                'cb902d3e3a4f475672398d24f228fd9307f8e8d9098642ea4b8fb89473bfd9f8',
                '5e78fd3c7d62f86ef1d0cd3df67b2e0085feb75934bd739652305d361fc6e6f7')}
    if increment==5:sources.add(('84f833db7b152264003c26b236c5c628c64b8773',None))
    if (p.get('code_commit'),p.get('source_digest')) not in sources:
        raise ValueError(f'{label} baseline unregistered source: path={path}, source_digest={p.get("source_digest")!r}, code_commit={p.get("code_commit")!r}')
    expected = dict(arm='icarl', seed=seed, backbone='resnet32_nobn', class_order_kind='superclass',
        policy_version='schur_gamma_bound_rho25_v2', algorithm_version='v4',
        upstream_commit=main.cil.UPSTREAM_PIN, fake_data=False, expected_tasks=100//increment,
        preflight_only=False, nobn_lr_scale=.5, coefficient=1., solver='gram', curvature='power10',
        gir_live=False,
        dataset_fingerprint=dataset_fingerprint(data_dir),
        class_order=main.class_order(seed,'superclass',data_dir))
    if increment!=5:expected.update(init_cls=increment,increment=increment)
    for key, value in expected.items():
        if p.get(key) != value: raise ValueError(label+' baseline mismatch: '+key)
    cfg = main.recipe(seed,'cuda:0','resnet32_nobn',increment)
    cfg['shuffle'] = False
    if p.get('cfg') != json.loads(json.dumps(cfg)):
        raise ValueError(label+' baseline mismatch: cfg')
    training = p.get('training', {})
    if (training != dict(first_epochs=100,later_epochs=50,first_milestones=[30,60,85],later_milestones=[24,35])
            or p.get('warmup_epochs',0) != 0 or p.get('grad_clip') is not None):
        raise ValueError(label+' baseline mismatch: training/warmup/clipping')
    native = path/'run' if (path/'run'/'protocol.json').is_file() else path
    saved_protocol = read(native/'protocol.json', {})
    tasks = summary.get('tasks', [])
    count = tasks[-1].get('parameter_count') if tasks else None
    # persist() adds observed model size and baseline mu0 to the immutable identity.
    expected_protocol = {**p, 'parameter_count': count, 'mu0': None}
    if not isinstance(count, int) or count <= 0 or saved_protocol != expected_protocol:
        mismatches = {k:dict(saved=saved_protocol.get(k), required=expected_protocol.get(k))
                      for k in saved_protocol.keys() | expected_protocol.keys()
                      if saved_protocol.get(k) != expected_protocol.get(k)
                      or (k in saved_protocol) != (k in expected_protocol)}
        raise ValueError(label+' baseline summary/protocol mismatch: '+json.dumps(mismatches,sort_keys=True))
    measurement = read(path/'measurement_protocol.json', {})
    if (measurement or increment==20) and (any(measurement.get(k) != p.get(k) for k in ('code_commit','source_digest','arm','seed','backbone'))
            or measurement.get('increment') != increment
            or measurement.get('selected_model') != 'final_epoch'
            or measurement.get('ggn',{}).get('definition') != 'engine_stability_current_minibatch_v2'):
        raise ValueError(label+' baseline measurement protocol mismatch')
    completed_tasks = divergence['task']-1 if failed_reference else 100//increment
    if (len(tasks) != completed_tasks or [row.get('task') for row in tasks] != list(range(1,completed_tasks+1))
            or read(native/'tasks.json', []) != tasks):
        raise ValueError(label+' baseline is missing tasks')
    for row in tasks:
        if any(not math.isfinite(row.get(k,float('nan'))) for k in ('cnn_top1','nme_top1')):
            raise ValueError(label+' baseline has nonfinite metrics')
    return dict(path=str(path),source_digest=p['source_digest'],code_commit=p['code_commit'],
        seed=seed,status='VALIDATED_ORIGINAL_BASELINE',outcome=outcome,diverged=copy.deepcopy(divergence),
        summary_path=str(summary_path),summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((native/'protocol.json').read_bytes()).hexdigest(),
        dataset_fingerprint=p['dataset_fingerprint'],
        baseline_measurements='recorded' if measurement else 'not_recorded_in_legacy_run')


def cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def live_state(hook):
    return cpu(dict(model=hook.learner._network.state_dict(),optimizer=hook.optimizer.state_dict(),
        gradients=[p.grad for p in hook.learner._network.parameters()],
        modes=[m.training for m in hook.learner._network.modules()],rng=rng_state(),
        steps=hook.steps,epoch_step=hook.epoch_step,
        intervention=None if hook.intervention is None else hook.intervention.state_dict()))


def curvature(hook,batch,r,seed):
    before=live_state(hook)
    started=time.monotonic()
    with _global_state(), torch.enable_grad():
        learner=copy.copy(hook.learner)
        learner._network=copy.deepcopy(learner._network)
        learner._old_network=copy.deepcopy(learner._old_network)
        model=main.Prediction(learner._network)
        spec,ctx=main.loss_context(learner,batch)
        engine=V3Engine('vna_gir',seed=seed,solver='gram',curvature='power10')
        engine.begin_task(learner._cur_task+1)
        stability=engine._stability(model,[],batch,spec,ctx,hook.steps,None)
        value=stability['lambda_max_ggn']
    _assert_bitwise(before,live_state(hook))
    return dict(task=learner._cur_task+1,r=r,iteration=hook.steps,lambda_max_ggn=value,
        finite=value is not None and math.isfinite(value),method='power10',window_size=len(batch[0]),
        generator_seed=stability['generator_seed'],loss='CE + native iCaRL KD (T=2)',
        known_classes=learner._known_classes,old_samples=int((batch[1]<learner._known_classes).sum()),
        window='current native training minibatch (new data + exemplar dataset)',phase='post_SGD_pre_intervention' if r else 'task_start',
        state_bitwise_unchanged=True,seconds=time.monotonic()-started)


def run(args):
    if not args.fake_data and not re.fullmatch('[0-9a-f]{64}',os.environ.get('VNA_SOURCE_DIGEST','')):
        raise ValueError('P1_COMPARE real runs require the SHA-256 verified source_digest')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    measurement=dict(code_commit=main.code_commit(),source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
        arm=args.arm,seed=args.seed,backbone=args.backbone,increment=args.increment,
        selected_model='final_epoch',tracker=dict(first_task=1,last_task=100//args.increment,
            dense_until=10,every=5,until=200,then='every_epoch_and_task_end'),
        ggn=dict(points=POINTS,iterations=10,window=4 if args.fake_data else 128,copy=True,bitwise_check=True,
            definition='engine_stability_current_minibatch_v2',loss='CE+KD',phase='post_SGD_pre_intervention'),
        fake_data=args.fake_data,fixture='two-channel residual, two one-epoch tasks' if args.fake_data else None)
    metadata=out/'measurement_protocol.json'
    if metadata.exists() and read(metadata)!=json.loads(json.dumps(measurement)):
        raise ValueError('measurement resume identity mismatch')
    if not metadata.exists() and (out/'run'/'summary.json').exists():
        raise ValueError('existing uninstrumented run cannot authorize this measurement protocol')
    write(metadata,measurement)
    tracker=Tracker(out,torch,np,min_task=1,max_tasks=100//args.increment)
    path=out/'ggn_power10.json';rows=read(path,[])
    original_before=main.MainHook.before_optimizer_step
    original_record=main.MainHook.record_task
    original_after=main.MainHook.after_optimizer_step
    original_intervene=main.Intervention.after_step
    def measure(hook,r):
        task=hook.learner._cur_task+1
        if r not in POINTS or any(row['task']==task and row['r']==r for row in rows):return
        batch=tuple(t.to(hook.learner._device) for t in hook.current_batch)
        if args.fake_data:batch=tuple(t[:4] for t in batch)
        rows.append(curvature(hook,batch,r,args.seed));write(path,rows)
        main.log(f'GGN task={task} r={r}: {json.dumps(rows[-1])}')
    def before(hook,frame):
        original_before(hook,frame)
        if args.preflight_only:return
        tracker.before_step(hook)
        task=hook.learner._cur_task+1
        r=hook.steps-1-tracker.state['tasks'][str(task)]['offset']
        if r==0:measure(hook,r)
    def intervene(intervention,hook,learner,optimizer):
        if not args.preflight_only:
            task=learner._cur_task+1
            measure(hook,hook.steps-tracker.state['tasks'][str(task)]['offset'])
        return original_intervene(intervention,hook,learner,optimizer)
    def record(hook,learner,result):
        if not args.preflight_only:
            tracker.task_end(hook,learner)
        value=original_record(hook,learner,result)
        if not args.preflight_only and hook.tasks and hook.tasks[-1]['task']==hook.manager.nb_tasks and 'feature_accuracy' not in hook.tasks[-1]:
            hook.tasks[-1]['feature_accuracy']=main.feature_accuracy(learner,hook.manager,hook.measurement_boundary)
            hook.persist(hook)
        return value
    def after(hook,frame):
        original_after(hook,frame)
        from experiments.grad_clip import check_parameters
        check_parameters(hook.learner,task=hook.learner._cur_task+1,epoch=hook.current_epoch+1,step=hook.steps)
        if not args.preflight_only and hook.intervention is None:
            task=hook.learner._cur_task+1
            measure(hook,hook.steps-tracker.state['tasks'][str(task)]['offset'])
    main.MainHook.before_optimizer_step=before;main.MainHook.record_task=record
    main.MainHook.after_optimizer_step=after
    main.Intervention.after_step=intervene
    original_out=args.out;args.out=str(out/'run')
    try:
        with ExitStack() as stack:
            if args.fake_data:
                from unittest.mock import patch
                from adapters.cil_setup import FixtureBackbone, skipinit
                main.cil.upstream_imports()
                from convs.cifar_resnet import ResNetBasicblock
                from utils import inc_net
                stack.enter_context(patch.object(main,'resnet32_nobn',lambda:skipinit(FixtureBackbone(ResNetBasicblock))))
                stack.enter_context(patch.object(inc_net,'get_convnet',lambda *a,**kw:FixtureBackbone(ResNetBasicblock)))
                native_after=main.Intervention.after_step
                def small_window(intervention,hook,learner,optimizer):
                    batch=hook.current_batch
                    hook.current_batch=tuple(t[:4] for t in batch)
                    try:return native_after(intervention,hook,learner,optimizer)
                    finally:hook.current_batch=batch
                stack.enter_context(patch.object(main.Intervention,'after_step',small_window))
            result=main.run(args)
    finally:
        main.MainHook.before_optimizer_step=original_before;main.MainHook.record_task=original_record
        main.MainHook.after_optimizer_step=original_after
        main.Intervention.after_step=original_intervene
        args.out=original_out
    events=read(out/'run'/'interventions.json',[])
    write(out/'intervention_log.json',[dict(event,
        accepted=bool((event.get('swaps') or 0)>0 or
                      (not event.get('skipped') and not event.get('rejected') and (event.get('s') or 0)>0)),
        h=event.get('h_dot_d'),swap=event.get('swaps',0)) for event in events])
    write(out/'summary.json',result)
    return result


def estimate_lane(increment,baseline_seconds,schedule,preflight_seconds):
    """Gram planning range: quadratic matrix work through cubic factorization."""
    points=schedule['calibration']
    if not points:raise ValueError('timing estimate requires measured preflight points')
    k0=int(schedule['K0'])
    if k0<1:raise ValueError('timing estimate requires positive K0')
    tasks=[]
    for task in range(2,100//increment+1):
        classes=task*increment
        updates=math.ceil((500*increment+2000)/128)*50
        steps=[1];gap=k0
        while steps[-1]+gap<=updates:
            steps.append(steps[-1]+gap);gap*=2
        estimates=[]
        for exponent in (2,3):
            costs=[(float(p['vna_seconds'])+float(p.get('stability_seconds',0))) *
                   (classes/(int(p['task'])*increment))**exponent for p in points]
            estimates.append(max(costs))
        tasks.append(dict(task=task,classes=classes,training_updates=updates,
            planned_applications=len(steps),application_steps=steps,
            seconds_per_application_range=estimates,
            intervention_seconds_range=[len(steps)*v for v in estimates]))
    intervention=[sum(row['intervention_seconds_range'][i] for row in tasks) for i in (0,1)]
    totals=[2*baseline_seconds+preflight_seconds+value for value in intervention]
    return dict(baseline_seconds=baseline_seconds,preflight_seconds=preflight_seconds,
        full_training_and_measurements_seconds=baseline_seconds,
        full_intervention_seconds_range=intervention,total_hours_range=[v/3600 for v in totals],
        sessions_11h_range=[math.ceil(v/(11*3600)) for v in totals],by_task=tasks,
        assumptions='Full training/Tracker/GGN/task-eval cost equals measured baseline. Intervention cost scales as classes^2..classes^3 from preflight; this is a planning range, not a bound. Geometric K0 cadence omits budget-guard extra doubling, which can reduce application count. Shared preflight is charged to both lanes as compute/wait reservation.')


if __name__=='__main__':run(main.parser().parse_args())
