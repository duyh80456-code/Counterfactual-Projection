"""Decisive VNA v3 pilot through pinned native training loops."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pickle
import random
import time
import numpy as np
import torch
from experiments import stage_b5 as lop
from experiments.vna_resume import Progress, Status, rng_state
from experiments.vna_provenance import code_commit
from adapters.lop_hooks import LearnHook, learner_state
from adapters.vna_v3_hooks import LoPV3, ICaRLV3
from diagnostics.vna_align import MeasurementBuffer

B5_ARMS = tuple('b5_'+a for a in ('bp', 'bp_kself', 'bp_vna', 'bp_vna_rand',
    'bp_gir', 'bp_vna_gir', 'cbp', 'cbp_gir', 'cbp_gradmax', 'cbp_vna_gir',
    'bp_vna_lf', 'bp_kself_lf'))
PM_ARMS = tuple('pm_'+a for a in ('bp', 'bp_kself', 'bp_vna', 'bp_vna_rand',
    'cbp', 'cbp_gir', 'cbp_gradmax', 'cbp_vna_gir'))
ICARL_ARMS = ('icarl', 'icarl_kself', 'icarl_vna_b0', 'icarl_vna', 'icarl_vna_rand')
ONLINE_ARMS = ('er', 'er_kself', 'er_vna', 'er_vna_rand', 'erace', 'erace_vna')


def engine_arm(arm):
    for prefix in ('b5_', 'pm_'):
        if arm.startswith(prefix): arm = arm[len(prefix):]
    if arm in ('bp', 'cbp', 'icarl', 'er', 'erace'): return 'baseline'
    arm = arm.removeprefix('bp_').removeprefix('icarl_').removeprefix('er_').removeprefix('erace_')
    return arm.replace('kself', 'kernel_self')


def read_schedule(args):
    if not args.schedule: return None
    schedule = json.loads(Path(args.schedule).read_text())
    expected = dict(setting=args.setting, code_commit=code_commit(),
                    source_digest=os.environ.get('VNA_SOURCE_DIGEST'))
    if args.setting == 'online': expected['model_config'] = online_model_config(args)
    for key, value in expected.items():
        if schedule.get(key) != value: raise ValueError(f'schedule {key} mismatch')
    if not schedule.get('authoritative') or not isinstance(schedule.get('K_base'), int) or schedule['K_base'] < 1:
        raise ValueError('schedule lacks authoritative positive K_base')
    if schedule.get('synthetic', False) and not args.fake_data:
        raise ValueError('synthetic calibration cannot authorize real training')
    return schedule


def make_engine(args):
    from methods.vna_v3 import V3Engine
    schedule = read_schedule(args)
    arm = engine_arm(args.arm)
    engine = V3Engine(arm=arm, seed=args.seed,
        coefficient=0. if arm == 'baseline' else args.coefficient,
        K_base=schedule['K_base'] if schedule else 100)
    if schedule: engine.configure_schedule(schedule)
    return engine


def online_model_config(args):
    return dict(nf=args.online_nf, n_tasks=args.online_tasks, mem_size=args.online_memory)


def protocol(args, engine, params, upstream):
    return dict(algorithm_version='v3', setting=args.setting, arm=args.arm, seed=args.seed,
        code_commit=code_commit(), source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
        upstream_commit=upstream, cfg=params, synthetic=args.fake_data,
        preflight_only=args.preflight_only, replacement_rate=params.get('replacement_rate'),
        replacement_rate_checks=[1e-6,1e-5,1e-4,1e-2] if args.setting == 'regression' else [],
        window=('current_augmented_minibatch' if args.setting == 'icarl' else
                'current augmented stream10 + native replay up to64; 74 after memory warmup' if args.setting == 'online' else
                'FIFO 128 most recent observed samples; available prefix at start; spans task boundaries'),
        step_rule=('min(s_star, K*lr/(1-momentum))' if args.arm.endswith('_lf') else
                   'min(s_star, lr*norm(g)/norm(d))'), K_units='iterations',
        mu_rel=engine.mu_rel, K_base=engine.K_base,
        solver=dict(method='parameter-space matrix-free CG', tol=engine.tol, max_iterations=50,
                    warm_start=True, spectral_power_iterations=engine.power_iters),
        preflight=read_schedule(args))


def finalize_calibration(args, engine, adapter, out):
    from methods.vna_v3 import schedule_from_points
    if len(adapter.calibration) != 6:
        raise ValueError('preflight requires three actual applications in tasks 1 and 2')
    accepted = []
    for mu in (.001,.01,.1):
        rows = [next((r for r in report['trials'] if r['mu_rel'] == mu),None)
                for report in adapter.mu_trials]
        if all(row is not None and row.get('Delta_hat',float('inf')) <=
               .1*abs(row.get('g_dot_d',0.))/max(row.get('gradient_norm',0.),1e-300)
               for row in rows): accepted.append(mu)
    engine.mu_rel = min(accepted) if accepted else .1
    for point, report in zip(adapter.calibration, adapter.mu_trials):
        selected = next(row for row in report['trials'] if row['mu_rel'] == engine.mu_rel)
        point['actual_application_seconds'] = point['vna_seconds']
        point['selected_mu_trial_seconds'] = selected['seconds']
        point['vna_seconds'] = selected['seconds']+point.get('gir_seconds',0.)
    timing = float(np.mean(adapter.iteration_times[:200]))
    points = []
    for task in (1, 2):
        values = [p for p in adapter.calibration if p['task'] == task]
        points.append(dict(task=task, classes=values[0]['classes'],
            vna_seconds=float(np.mean([p['vna_seconds'] for p in values])),
            iteration_seconds=timing))
    epoch_iters = adapter.full_batches[-1]
    schedule = schedule_from_points(points, epoch_iters)
    schedule.update(setting=args.setting, authoritative=True, authority_arm=args.arm,
        code_commit=code_commit(), source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
        synthetic=args.fake_data, mu_rel=engine.mu_rel, cg_tol=engine.tol,
        power_iters=engine.power_iters, tol=engine.tol, calibration=adapter.calibration,
        mu_trials=adapter.mu_trials, mu_condition_met=bool(accepted),
        timing_basis='selected-mu read-only trial plus measured GIR component; mean of first three points per task',
        iteration_samples=len(adapter.iteration_times[:200]),
        cos_cg_exact=[r.get('cos_cg_exact') for r in engine.events if r.get('cos_cg_exact') is not None])
    if args.setting == 'online': schedule['model_config'] = online_model_config(args)
    # B5/PM have fixed output width; only CIL extrapolates the class count.
    if args.setting in ('regression', 'mnist'):
        costs = max(p['vna_seconds'] for p in points)
        choices = sorted(set([50,100,200,400,800]+[n*epoch_iters for n in (1,2,5,10)]))
        feasible = [k for k in choices if costs/k <= timing]
        schedule.update(K_base=min(feasible) if feasible else max(choices),
                        budget_exceeded=not bool(feasible), projected_vna_seconds=costs,
                        projection='fixed output width; no class extrapolation')
    path = Path(args.schedule_output) if args.schedule_output else out/'schedule.json'
    lop.write_json(path, schedule)
    print('V3 PREFLIGHT', json.dumps(schedule), flush=True)
    return schedule


def write_events(out, engine):
    lop.write_json(out/'interventions.json', engine.events)
    if engine.events: lop.write_csv(out/'vna_log.csv', engine.events)
    else: (out/'vna_log.csv').write_text('task,iteration,K,K_next,rho,seconds,skipped\n')


def lop_recipe(args):
    lop.upstream_imports()
    base = 'cbp' if '_cbp' in args.arm else 'bp'
    rate = args.replacement_rate if base == 'cbp' else None
    params, source = lop.recipe(args.setting, base, rate)
    if args.setting == 'mnist':
        # Restore the complete upstream horizon; Stage B intentionally truncated it.
        cfg = json.loads((lop.UPSTREAM/source['cfg_source']).read_text())
        params.update(num_tasks=cfg['num_tasks'], num_features=10, num_hidden_layers=3)
    params.update(run_idx=0, use_gpu=int(args.device.startswith('cuda')))
    if args.preflight_only:
        if args.setting == 'regression': params['num_data_points'] = 2*int(params['flip_after'])
        else: params['num_tasks'] = 2
    if args.fake_data:
        if args.setting == 'regression': params.update(num_data_points=400, flip_after=200)
        else: params.update(num_tasks=2, change_after=200, fixture_rank_padding=True)
    return base, params, source


def run_lop(args):
    base, params, provenance = lop_recipe(args)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    params['data_file'] = str(out/'upstream_in_progress.tmp')
    if args.setting == 'regression': params['env_file'] = str(out/'data/regression.pkl')
    if args.device.startswith('cuda'): torch.cuda.set_device(args.device)
    if args.fake_data: torch.set_num_threads(1)
    engine = make_engine(args)
    identity = protocol(args, engine, params, lop.UPSTREAM_PIN)
    identity['expected_tasks'] = (int(params['num_data_points'])//int(params['flip_after'])
                                  if args.setting == 'regression' else int(params['num_tasks']))
    identity.update(coefficient=args.coefficient)
    manifest = out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('v3 resume identity mismatch')
    lop.write_json(manifest, identity); lop.write_json(out/'protocol.json', identity)
    if args.resume and (out/'summary.json').exists():
        previous = json.loads((out/'summary.json').read_text())
        if previous.get('status') == 'COMPLETED': return previous
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    lop.prepare_data(args.setting, params, out, fake=args.fake_data, data_dir=args.data_dir)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    intervention = LoPV3(engine, benchmark=args.setting, preflight=args.preflight_only)
    checkpoint_path = out/'resume.pkl'
    resumed = None
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open('rb') as stream: resumed = pickle.load(stream)
        if resumed.get('identity') != identity: raise ValueError('v3 checkpoint identity mismatch')
    def checkpoint(state):
        lop.atomic_pickle(checkpoint_path, {**state, 'identity': identity})
        write_events(out, engine)
        lop.write_json(out/'tasks.json', list(hook.results.tasks()))
        return checkpoint_path.stat().st_size
    status = Status(out, f'v3 {args.arm} seed={args.seed}')
    hook = LearnHook(args.setting, base, params, seed=args.seed, checkpoint=checkpoint,
        resume=resumed, intervention=intervention,
        progress=Progress(out, identity, name='deadline.pt'), status=status)
    hook.task_log_only = True
    hook.buffer = MeasurementBuffer(128)
    started = time.monotonic()
    try:
        with intervention.install(): logs = lop.run_native(args.setting, params, out, hook)
        hook._checkpoint(hook.last_learner, hook.last_frame)
        lop.atomic_pickle(out/'final_state.pkl', dict(learner=learner_state(hook.last_learner), rng=rng_state()))
        lop.atomic_pickle(out/'upstream.pkl', logs)
        by_task = {r['task']:r for r in intervention.task_metrics}
        tasks = [{**row, **by_task.get(row['task'], {})} for row in hook.results.tasks()]
        seconds = hook.elapsed_base+time.monotonic()-started
        events_seconds = sum(e.get('seconds',0.) for e in engine.events)
        summary = dict(algorithm_version='v3', setting=args.setting, arm=args.arm, seed=args.seed,
            status='COMPLETED', tasks=tasks, tasks_run=len(tasks), seconds=seconds,
            code_commit=code_commit(), source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
            replacement_rate=params['replacement_rate'], steps=hook.step,
            intervention_seconds=events_seconds, training_seconds=max(0.,seconds-events_seconds),
            swaps=sum(e.get('swaps',0) for e in engine.events),
            skip_rate=sum(bool(e.get('skipped')) for e in engine.events)/max(1,len(engine.events)),
            anytime_loss_auc=float(np.mean([t['anytime_loss_auc'] for t in tasks])),
            identity=identity)
        if args.setting == 'regression':
            summary['mean_last50'] = float(np.mean([t['online_error'] for t in tasks[-50:]]))
        else:
            summary['mean_last_half_accuracy'] = float(np.mean([t['online_accuracy'] for t in tasks[len(tasks)//2:]]))
            summary['anytime_accuracy_auc'] = float(np.mean([t['anytime_accuracy_auc'] for t in tasks]))
        if args.preflight_only:
            identity.update(preflight=finalize_calibration(args,engine,intervention,out))
        identity.update(mu_rel=engine.mu_rel, K_base=engine.K_base)
        lop.write_json(out/'protocol.json', identity)
        lop.write_json(out/'summary.json', summary); lop.write_json(out/'tasks.json', tasks)
        lop.write_csv(out/'online.csv', tasks); lop.write_json(out/'activities.json', engine.activities)
        lop.write_json(out/'unit_contributions.json', intervention.contributions)
        write_events(out, engine)
        return summary
    finally: status.close()


def run_icarl(args):
    from experiments import stage_b3
    from diagnostics.icarl_measure import B3Measurement
    stage_b3.upstream_imports()
    engine = make_engine(args)
    intervention = ICaRLV3(engine, preflight=args.preflight_only)
    native_args = stage_b3.parser().parse_args([])
    for name in ('out','device','data_dir','seed','workers','fake_data'):
        setattr(native_args, name, getattr(args,name))
    native_args.init_seed = 0
    native_args.arm = args.arm
    native_args.smoke = args.fake_data or args.preflight_only
    native_args.measure_profile = 'quick'
    native_args.no_measure = True
    class NoMeasurement(B3Measurement):
        def __call__(self,*a,**kw): return []
    measure = NoMeasurement(dict(setting='icarl',arm=args.arm),measure_profile='quick')
    original_runtime = stage_b3.runtime
    @contextmanager
    def runtime(**kw):
        with original_runtime(**kw):
            if args.preflight_only:
                from models import icarl
                icarl.init_epoch = icarl.epochs = 4 if args.fake_data else 5
            yield
    stage_b3.runtime = runtime
    try:
        hook = stage_b3.run(native_args,measure_override=measure,intervention=intervention,
                            observe=intervention.after_epoch)
    finally: stage_b3.runtime = original_runtime
    out = Path(args.out)
    info = protocol(args,engine,stage_b3.recipe(),stage_b3.UPSTREAM_PIN)
    info['expected_tasks'] = 2 if args.fake_data or args.preflight_only else 10
    if args.preflight_only: info['preflight'] = finalize_calibration(args,engine,intervention,out)
    lop.write_json(out/'protocol.json',info)
    lop.write_csv(out/'continual_eval.csv',intervention.continual_eval)
    summary = json.loads((out/'summary.json').read_text())
    info['dataset_fingerprint'] = summary.get('identity',{}).get('dataset_fingerprint')
    if not args.fake_data and info['dataset_fingerprint'] is None:
        from experiments.parallel_validity import dataset_fingerprint
        info['dataset_fingerprint'] = dataset_fingerprint(out/'data')
    lop.write_json(out/'protocol.json',info)
    summary.update(algorithm_version='v3', setting='icarl', anytime_accuracy_auc=None,
                   code_commit=code_commit(),source_digest=os.environ.get('VNA_SOURCE_DIGEST'))
    for task in summary['tasks']:
        curves = [v for v in intervention.continual_eval if v['task'] == task['task']]
        losses = intervention.task_losses.get(task['task'],[])
        half = [v['iteration'] for v in losses if v['loss'] <= losses[0]['loss']*.5]
        if curves:
            x = np.array([v['iteration'] for v in curves]); y = np.array([v['accuracy'] for v in curves])
            old = [v['old_accuracy'] for v in curves if v['old_accuracy'] is not None]
            reference = summary['tasks'][task['task']-2]['cnn_top1'] if task['task'] > 1 else None
            recovered = [v for v in curves if reference is not None and v['old_accuracy'] is not None and v['old_accuracy'] >= reference]
            task.update(learning_accuracy=curves[-1]['new_accuracy'], old_accuracy=curves[-1]['old_accuracy'],
                        new_accuracy=curves[-1]['new_accuracy'],
                        anytime_accuracy_auc=float(np.trapezoid(y,x)/max(1,x[-1]-x[0])) if len(x)>1 else float(y[0]),
                        min_old_accuracy=min(old) if old else None,
                        stability_gap_depth=max(0.,reference-min(old)) if old and reference is not None else None,
                        stability_recovery_iterations=recovered[0]['iteration']-curves[0]['iteration'] if recovered else None,
                        iterations_to_half_loss=half[0]-losses[0]['iteration'] if half else None)
    if summary['tasks']:
        summary['anytime_accuracy_auc'] = float(np.mean([t.get('anytime_accuracy_auc',0) for t in summary['tasks']]))
    lop.write_json(out/'summary.json',summary); lop.write_json(out/'tasks.json',summary['tasks'])
    write_events(out,engine)
    return summary


def run(args):
    allowed = dict(regression=B5_ARMS,mnist=PM_ARMS,icarl=ICARL_ARMS,online=ONLINE_ARMS)
    if args.arm not in allowed[args.setting]: raise ValueError('arm does not belong to setting')
    if args.preflight_only and args.arm not in ('b5_bp_vna_gir','pm_bp_vna','icarl_vna','er_vna'):
        raise ValueError('preflight authority must use the registered most expensive VNA arm')
    if not args.preflight_only and engine_arm(args.arm) != 'baseline' and not args.schedule and not args.fake_data:
        raise ValueError('full arm requires a shared authoritative family schedule')
    if args.setting == 'online':
        from experiments.pilot_v3_online import run_online
        return run_online(args)
    return run_icarl(args) if args.setting == 'icarl' else run_lop(args)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setting',choices=('regression','mnist','icarl','online'),required=True)
    p.add_argument('--arm',required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',required=True); p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cpu'); p.add_argument('--data-dir')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--replacement-rate',type=float,choices=lop.RATES,default=1e-3)
    p.add_argument('--schedule'); p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--schedule-output'); p.add_argument('--fake-data',action='store_true')
    p.add_argument('--coefficient',type=float,default=1.)
    p.add_argument('--online-nf',type=int,choices=(20,64),default=64)
    p.add_argument('--online-tasks',type=int,choices=(10,20),default=10)
    p.add_argument('--online-memory',type=int,choices=(1000,2000),default=2000)
    return p


if __name__=='__main__': run(parser().parse_args())
