"""Run the pinned LoP experiment functions unchanged, with read-only hooks."""
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import torch._decomp.decompositions_for_jvp  # before run_native patches torch.zeros
from experiments.vna_resume import log
from experiments.io_utils import working_directory, atomic_pickle, write_json, write_csv

ROOT = Path(__file__).resolve().parents[2]
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
                    'recipe_decision': 'registered native LoP recipe (2026-10-15)',
                    'step_size': lr, 'replacement_rate': params['replacement_rate'],
                    'momentum': params.get('momentum', 0), 'effective_maturity': params['gir_maturity']}


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

