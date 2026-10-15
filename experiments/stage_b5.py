"""Run the pinned LoP experiment functions unchanged, with read-only hooks."""
import argparse
from experiments.vna_resume import configure_measurement, measurement_cli
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import torch._decomp.decompositions_for_jvp  # before run_native patches torch.zeros
from experiments.vna_resume import Progress, Status, log, rng_state, restore_rng

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'third_party/loss-of-plasticity'
UPSTREAM_PIN = 'a6b79580d85f3025bdb601566d3627c5f489f13b'
RATES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
CFG = {'regression': ('slowly_changing_regression', 'sgd/bp/relu', 'sgd/cbp/relu', .01),
       'mnist': ('permuted_mnist', 'bp/std_net', 'cbp', .003),
       'imagenet': ('imagenet', 'bp', 'cbp', .01)}


def upstream_imports():
    if not (UPSTREAM/'lop').is_dir():
        raise FileNotFoundError('pinned loss-of-plasticity absent (no automatic fetch)')
    commit = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_PIN: raise ValueError('loss-of-plasticity commit does not match the registered pin')
    if str(UPSTREAM) not in sys.path: sys.path.insert(0, str(UPSTREAM))


def recipe(benchmark, arm, rate=None):
    folder, bp, cbp, lr = CFG[benchmark]
    source = f'lop/{folder}/cfg/{bp if arm == "bp" else cbp}.json'
    params = json.loads((UPSTREAM/source).read_text())
    params.update(step_size=lr, agent=arm)
    if benchmark == 'regression':
        if arm == 'cbp' and rate not in RATES: raise ValueError('regression CBP requires a pre-registered replacement rate')
        params['replacement_rate'] = 0 if arm == 'bp' else rate
    elif benchmark == 'mnist':
        params['replacement_rate'] = 0 if arm == 'bp' else 1e-4
        params['num_tasks'] = 200
    else:
        params['replacement_rate'] = 0 if arm == 'bp' else 3e-4
        params['num_tasks'] = 200
    params['gir_replacement_rates'] = list(RATES) if benchmark == 'regression' else [3e-4 if benchmark == 'imagenet' else 1e-4]
    # expr reads mt, not maturity_threshold, and defaults to ten.
    params['gir_maturity'] = params.get('mt', 10) if benchmark == 'regression' else params.get('mt', params.get('maturity_threshold', 100))
    return params, {'cfg_source': source, 'step_size_source': f'lop/{folder}/cfg/{cbp}.json',
                    'recipe_decision': 'TASK_STAGE_B5.md §5 (2026-10-15)',
                    'step_size': lr, 'replacement_rate': params['replacement_rate'],
                    'momentum': params.get('momentum', 0), 'effective_maturity': params['gir_maturity']}


@contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(path)
    try: yield
    finally: os.chdir(old)


def atomic_pickle(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f: pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, default=str)+'\n'); tmp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = Path(path).with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def merge_json(path, key, value):
    """Serialize the independent GPU workers' gate updates."""
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {}
        data[key] = value; write_json(path, data)


def restore_archives(input_root, output):
    from experiments.vna_kaggle import restore_stage_archives
    return restore_stage_archives(input_root, output, 'stageB')


def prepare_data(benchmark, params, run_dir, *, fake=False, smoke=False, data_dir=None):
    """Stage local inputs; real MNIST always uses the original pickle builder."""
    (run_dir/'data').mkdir(exist_ok=True)
    gen = torch.Generator().manual_seed(871)
    if benchmark == 'regression':
        params['env_file'] = str(run_dir/'data/regression.pkl')
        if fake:
            x = torch.randint(2, (int(params['num_data_points']), params['num_inputs']), generator=gen).float()
            y = x.sum(1, keepdim=True)/params['num_inputs']
            atomic_pickle(params['env_file'], (x, y, None))
        elif not Path(params['env_file']).exists():
            from lop.slowly_changing_regression.slowly_changing_regression import generate_problem_data
            # Use their generator unchanged. It writes one extra flip interval
            # and processes 10k chunks; full recipe has valid 10k intervals.
            generate_problem_data(data_file=params['env_file'], flip_after=params['flip_after'],
                                  num_data_points=max(10000, params['num_data_points']), num_inputs=params['num_inputs'],
                                  num_target_features=params['num_target_features'], num_flipping_bits=params['num_flipping_bits'],
                                  beta=params['beta'])
    elif benchmark == 'mnist':
        dest = run_dir/'data/mnist_'
        if dest.exists(): return True
        if fake:
            # Upstream fixes the data permutation size at 60k. Expand a small
            # synthetic bank, preserving the 784-dimensional native interface.
            x = torch.rand(1, 784, generator=gen).expand(60000, -1)
            y = torch.arange(60000)%10
            atomic_pickle(dest, (x, y, x[:100], y[:100]))
        elif data_dir and (Path(data_dir)/'mnist_').exists():
            shutil.copy2(Path(data_dir)/'mnist_', dest)
        else:
            if data_dir:
                raw = Path(data_dir)
                if (raw/'MNIST').exists():
                    shutil.copytree(raw/'MNIST', run_dir/'data/MNIST', dirs_exist_ok=True)
                elif (raw/'raw').exists():
                    shutil.copytree(raw, run_dir/'data/MNIST', dirs_exist_ok=True)
                else: raise FileNotFoundError('MNIST_DIR must contain mnist_ or torchvision MNIST/raw')
            from lop.permuted_mnist.load_mnist import mnist
            with working_directory(run_dir): mnist()  # downloads only when explicitly run on Kaggle
    else:
        classes = run_dir/'data/classes'
        if fake:
            classes.mkdir(exist_ok=True)
            # Smoke uses the native minibatch of 100: 2 batches/task.
            n_train = params.get('fake_train_images_per_class', 600)
            n_test = params.get('fake_test_images_per_class', 100)
            for i in range(4):
                rng = np.random.RandomState(871+i)
                np.save(classes/f'{i}.npy', rng.standard_normal((n_train+n_test, 3, 32, 32)).astype(np.float32))
            order = [np.tile(np.arange(4), 250)]
            atomic_pickle(run_dir/'class_order', order)
        else:
            if not data_dir or not Path(data_dir).is_dir(): return False
            if not classes.exists(): classes.symlink_to(Path(data_dir).resolve(), target_is_directory=True)
            shutil.copy2(UPSTREAM/'lop/imagenet/class_order', run_dir/'class_order')
    return True


def run_native(benchmark, params, run_dir, hook):
    """Only fixture size constants change in fake/smoke tests; native loops do not."""
    from contextlib import nullcontext
    context = hook.install() if hook else nullcontext()
    with working_directory(run_dir), context:
        if benchmark == 'regression':
            from lop.slowly_changing_regression.expr import expr
            return expr(params)
        if benchmark == 'mnist':
            import lop.permuted_mnist.online_expr as module
            # Tiny sequences expose an upstream zero-sized rank log. Padding
            # its three rank-log allocations is a test-fixture adapter only;
            # neither samples, permutations nor the training steps are changed.
            original_rank = module.compute_matrix_rank_summaries
            rank_task = None
            def rank_progress(*args, **kwargs):
                nonlocal rank_task
                import inspect
                frame = inspect.currentframe().f_back
                task = frame.f_locals['task_idx']+1
                if hook and hook.progress:
                    hook.progress.boundary(lambda: hook._checkpoint(frame.f_locals['learner'], frame)
                                           if hook.resume is None else None)
                if hook and hook.resume is None and rank_task != task:
                    rank_task = task
                    if hook.status: hook.status.update(task=task, step=hook.step, phase='upstream rank SVD')
                    hook._checkpoint(frame.f_locals['learner'], frame)
                started = time.monotonic()
                log(f'MNIST task={task} upstream rank SVD start')
                result = original_rank(*args, **kwargs)
                log(f'MNIST task={task} upstream rank SVD seconds={time.monotonic()-started:.3f}')
                return result
            module.compute_matrix_rank_summaries = rank_progress
            original = module.torch.zeros
            if params.get('fixture_rank_padding'):
                def zeros(shape, *args, **kwargs):
                    if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == 0:
                        shape = (1, shape[1])
                    return original(shape, *args, **kwargs)
                module.torch.zeros = zeros
            try: result = module.online_expr(params)
            finally:
                module.torch.zeros = original
                module.compute_matrix_rank_summaries = original_rank
        else:
            import lop.imagenet.single_expr as module
            old = module.train_images_per_class, module.test_images_per_class, module.images_per_class
            if 'fake_train_images_per_class' in params:
                module.train_images_per_class = params['fake_train_images_per_class']
                module.test_images_per_class = params['fake_test_images_per_class']
                module.images_per_class = module.train_images_per_class+module.test_images_per_class
            try: result = module.repeat_expr(params)
            finally: module.train_images_per_class, module.test_images_per_class, module.images_per_class = old
        with open(params['data_file'], 'rb') as f: logs = pickle.load(f)
        return logs if result is None else result


def baseline_gate(benchmark, bp, cbps, *, tolerance=.02):
    """Fixed early/late rule; regression chooses the lowest mean run error."""
    error = benchmark == 'regression'
    key = 'online_error' if error else 'online_accuracy'
    if not bp or not cbps: return {'status': 'PENDING', 'reason': 'both arms required'}
    chosen = min(cbps, key=lambda arm: np.mean([r[key] for r in cbps[arm]])) if error else next(iter(cbps))
    def means(rows):
        values = [r[key] for r in sorted(rows, key=lambda r: r['task'])]
        return float(np.mean(values[:10])), float(np.mean(values[-10:])), float(np.mean(values))
    early, late, _ = means(bp); cearly, clate, mean = means(cbps[chosen])
    bp_declines = late > early if error else late < early
    cbp_holds = clate <= cearly+tolerance if error else clate >= cearly-tolerance
    sufficient = len(bp) >= 20 and len(cbps[chosen]) >= 20
    return {'status': 'PASS' if sufficient and bp_declines and cbp_holds else 'FAIL',
            'metric': key, 'bp_early': early, 'bp_late': late, 'cbp_early': cearly, 'cbp_late': clate,
            'cbp_mean': mean, 'cbp_selected_arm': chosen, 'tolerance': tolerance,
            'bp_declines': bp_declines, 'cbp_holds': cbp_holds, 'sufficient_tasks': sufficient,
            'early_window': 'tasks 1–10', 'late_window': 'last 10 tasks'}


def finalize(output, benchmark, *, mu_rel_default=.01):
    from diagnostics.vna_gate import evaluate_gate, setting_config
    base = output/'b5'/benchmark
    bp, cbps, measurements, tasks = [], {}, [], 0
    synthetic = False
    for path in base.glob('*/summary.json'):
        summary = json.loads(path.read_text())
        if summary.get('status') == 'PAUSED': continue
        synthetic |= summary.get('fake_data', False) or summary.get('smoke', False)
        if summary['arm'] == 'bp':
            bp = summary['tasks']; tasks = len(bp)
            rows_path = path.parent/'measurements.json'
            if rows_path.exists(): measurements = json.loads(rows_path.read_text())
        else: cbps[path.parent.name] = summary['tasks']
    if not bp and not cbps and benchmark == 'imagenet':
        path = output/'gate.json'
        if path.exists():
            entry = json.loads(path.read_text()).get('B5_imagenet')
            if entry and entry['status'] == 'skipped: data unavailable': return entry
    expected_cbps = 5 if benchmark == 'regression' else 1
    baseline = baseline_gate(benchmark, bp, cbps) if len(cbps) == expected_cbps else {'status': 'PENDING', 'reason': 'all registered CBP arms required'}
    if synthetic:
        baseline.update(status='FAIL', reason='synthetic/smoke runs cannot establish the baseline')
    baseline_path = output/'b5_baseline_gate.json'
    merge_json(baseline_path, benchmark, baseline)
    config = setting_config('B5', tasks_run=tasks, mu_rel_default=1e-2)
    # Regression's registered maximum rank is four, never sixteen.
    config['gate_rank'] = 4 if benchmark == 'regression' else 16
    gate = evaluate_gate(measurements, config)
    gate.update(baseline=baseline, measurements_valid=baseline['status'] == 'PASS', benchmark=benchmark)
    if not gate['measurements_valid']: gate['status'] = 'INVALID'
    path = output/'gate.json'
    merge_json(path, f'B5_{benchmark}', gate)
    return gate


def run(args):
    name = args.arm if args.replacement_rate is None else f'cbp_rr_{args.replacement_rate:g}'
    out = Path(args.out).resolve()/'b5'/args.benchmark/name
    out.mkdir(parents=True, exist_ok=True)
    status = Status(out, f'B5/{args.benchmark}/{name}')
    status.update(phase='preparing data')
    try:
        result = _run(args, status)
        status.update(phase='completed')
        return result
    finally:
        status.close()


def _run(args, status):
    timing_started = time.monotonic()
    upstream_imports()
    if args.benchmark == 'regression' and args.arm == 'cbp' and args.replacement_rate is None:
        from copy import copy
        summaries = []
        for rate in RATES:
            child = copy(args); child.replacement_rate = rate
            summaries.append(run(child))
        return summaries
    params, provenance = recipe(args.benchmark, args.arm, args.replacement_rate)
    actual_device = 'cpu' if args.benchmark == 'regression' else args.device
    params['use_gpu'] = int(actual_device.startswith('cuda'))
    if actual_device.startswith('cuda'): torch.cuda.set_device(torch.device(actual_device))
    if args.smoke:
        if args.benchmark == 'regression': params.update(num_data_points=1000, flip_after=500)
        else: params['num_tasks'] = 2
        if args.benchmark == 'mnist': params.update(change_after=500, fixture_rank_padding=True)
        if args.benchmark == 'imagenet':
            params['num_showings'] = 1
            if args.fake_data: params.update(fake_train_images_per_class=100, fake_test_images_per_class=50)
    arm_name = args.arm if args.replacement_rate is None else f'cbp_rr_{args.replacement_rate:g}'
    output = Path(args.out).resolve(); run_dir = output/'b5'/args.benchmark/arm_name
    run_dir.mkdir(parents=True, exist_ok=True)
    params['data_file'] = str(run_dir/'upstream_in_progress.tmp'); params['run_idx'] = 0
    if args.benchmark == 'regression': params['env_file'] = str(run_dir/'data/regression.pkl')
    params['examples_per_epoch'] = params.get('fake_train_images_per_class', 600)*params.get('num_classes', 2)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    signature = {'measurement_protocol': 'round2-nu', 'solver': getattr(args, 'solver', 'auto'),
                 'output_exact_max': getattr(args, 'output_exact_max', 4096), 'params': params, 'seed': args.seed, 'code_commit': commit,
                 'fake_data': args.fake_data, 'smoke': args.smoke, 'mu_rel_default': args.mu_rel_default,
                 'mu_source': args.mu_rel_default_source, 'mu_rel_default_source': args.mu_rel_default_source}
    manifest = run_dir/'manifest.json'
    previous = None
    if args.resume and (run_dir/'summary.json').exists():
        previous = json.loads((run_dir/'summary.json').read_text())
    expected_tasks = (int(params['num_data_points'])//int(params['flip_after'])
                      if args.benchmark == 'regression' else params['num_tasks'])
    completed = (previous is not None and previous.get('status') in (None, 'COMPLETED')
                 and len(previous.get('tasks', [])) == expected_tasks)
    if args.resume and manifest.exists():
        old_signature = json.loads(manifest.read_text())
        compatible = old_signature == signature
        if not compatible: raise ValueError('resume configuration/commit mismatch')
    elif completed:
        raise ValueError('completed B5 run has no manifest for identity validation')
    if completed:
        log(f'reusing completed {args.benchmark} {arm_name} code_commit={previous.get("code_commit")}')
        return previous
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not prepare_data(args.benchmark, params, run_dir, fake=args.fake_data, smoke=args.smoke, data_dir=args.data_dir):
        entry = {'status': 'skipped: data unavailable', 'measurements_valid': False}
        merge_json(output/'gate.json', f'B5_{args.benchmark}', entry)
        return entry
    # Data generation has its own deterministic setup, then both arms begin
    # with identical baseline streams, independent of preprocessing caches.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    signature['params'] = params; write_json(manifest, signature)
    metadata = {'schema_version': 'stage1d-v1', 'benchmark': args.benchmark, 'arm': arm_name,
                'null_kind': 'N2', 'n2_centered': args.benchmark != 'regression',
                'seed': args.seed, 'code_commit': commit, 'device': actual_device, 'fake_data': args.fake_data,
                'mu_rel_default': args.mu_rel_default, 'mu_source': args.mu_rel_default_source, 'mu_rel_default_source': args.mu_rel_default_source,
                **provenance, 'upstream_commit': UPSTREAM_PIN, 'cfg_values': json.dumps(params, sort_keys=True)}
    from adapters.lop_hooks import LearnHook
    from diagnostics.lop_measure import B5Measurement
    checkpoint_path = run_dir/'resume.pkl'
    resumed = None
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open('rb') as f: resumed = pickle.load(f)
        if resumed.get('identity', signature) != signature:
            raise ValueError('B5 checkpoint identity mismatch')
    progress = Progress(run_dir, signature, name='deadline.pt')
    status.update(phase='data ready')
    def checkpoint(state):
        atomic_pickle(checkpoint_path, {**state, 'identity': signature})
        write_json(run_dir/'measurements.json', state['rows'])
        write_csv(run_dir/'measurements.csv', state['rows'])
        return checkpoint_path.stat().st_size
    hook = LearnHook(args.benchmark, args.arm, params,
                     configure_measurement(B5Measurement(args.benchmark, metadata, mu_rel_default=args.mu_rel_default), args, run_dir),
                     seed=args.seed, checkpoint=checkpoint, resume=resumed,
                     time_steps=args.smoke, timing_started=timing_started, progress=progress, status=status)
    start = time.monotonic()
    try:
        logs = run_native(args.benchmark, params, run_dir, hook)
    except Exception as error:
        merge_json(output/'gate.json', f'B5_{args.benchmark}',
                   {'status': 'INVALID', 'measurements_valid': False, 'run_error': str(error)})
        raise
    finally:
        status.update(phase='saving checkpoint')
    # The native loop has now consumed its final learn return and updated logs.
    hook._checkpoint(hook.last_learner, hook.last_frame)
    hook.last_frame = None
    atomic_pickle(run_dir/'upstream.pkl', logs)
    Path(params['data_file']).unlink(missing_ok=True)
    tasks = [{**metadata, **task} for task in hook.results.tasks()]
    summary = {**metadata, 'status': 'COMPLETED', 'arm': args.arm, 'tasks': tasks, 'wall_seconds': time.monotonic()-start,
               'smoke': args.smoke, 'tasks_run': len(tasks), 'measurements': len(hook.rows)}
    summary['measurement_point_seconds'] = hook.measurement_point_seconds
    summary['measurement_seconds'] = sum(point['seconds'] for point in hook.measurement_point_seconds)
    summary['training_seconds'] = max(0., summary['wall_seconds']-summary['measurement_seconds'])
    summary['steps_run'] = hook.step
    if args.smoke and args.benchmark == 'mnist':
        # Time exactly 2000 plain upstream learn calls, with the hook absent.
        import copy
        from adapters.lop_hooks import detached
        live = hook.last_learner
        features = live.previous_features
        live.previous_features = detached(features)
        try: learner = copy.deepcopy(live)
        finally: live.previous_features = features
        device = next(learner.net.parameters()).device
        x, y = (part[-params.get('mini_batch_size', 1):].to(device) for part in hook.buffer.batch())
        saved_rng = rng_state()
        if device.type == 'cuda': torch.cuda.synchronize(device)
        started = time.monotonic()
        for _ in range(2000): learner.learn(x=x, target=y)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        upstream_seconds = (time.monotonic()-started)/2000
        restore_rng(saved_rng)
        summary['upstream_step_seconds'] = upstream_seconds
        hooked = float(np.median(hook.step_wall_seconds[hook.task_steps:]))
        log(f'MNIST 2000 hook-disabled steps upstream={upstream_seconds*1000:.3f} ms/step '
            f'hook={hooked*1000:.3f} ms/step overhead={(hooked-upstream_seconds)*1000:.3f} ms/step')
    if args.smoke:
        second_task = hook.step_wall_seconds[hook.task_steps:2*hook.task_steps]
        summary['startup_seconds'] = hook.startup_seconds
        summary['step_wall_seconds'] = hook.step_wall_seconds
        summary['steady_step_seconds'] = float(np.median(second_task))
    write_json(run_dir/'measurements.json', hook.rows); write_csv(run_dir/'measurements.csv', hook.rows)
    write_csv(run_dir/'online.csv', tasks); write_json(run_dir/'summary.json', summary)
    atomic_pickle(run_dir/'final_state.pkl', __import__('adapters.lop_hooks', fromlist=['learner_state']).learner_state(hook.last_learner))
    finalize(output, args.benchmark, mu_rel_default=args.mu_rel_default)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    measurement_cli(p)
    p.add_argument('benchmark', choices=CFG)
    p.add_argument('--arm', required=True, choices=['bp', 'cbp'])
    p.add_argument('--smoke', action='store_true'); p.add_argument('--fake-data', action='store_true')
    p.add_argument('--resume', action='store_true'); p.add_argument('--out', default='outputs/stageB')
    p.add_argument('--data-dir'); p.add_argument('--device', default='cpu'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mu-rel-default', type=float, default=.01)
    p.add_argument('--mu-rel-default-source', default='cli/default')
    p.add_argument('--replacement-rate', type=float, choices=RATES)
    return p


def main():
    args = parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
