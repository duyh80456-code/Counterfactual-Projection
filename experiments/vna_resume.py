"""Shared Stage-B checkpoint identity, isolated hook state and session deadlines."""
import copy
import ctypes
import hashlib
import os
from pathlib import Path
import random
import numpy as np
import shutil
import sys
import time
import threading
import torch
from experiments.shared_protocol import atomic_torch_save, atomic_json_save
from methods.vna import dedicated_generator


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    if 'numpy' in state: np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def log(message):
    print(f"[{os.environ.get('VNA_LANE', 'CPU')} {os.environ.get('VNA_JOB', 'stageB')}] {message}", flush=True)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_locals(frame, values):
    # Python >= 3.13 (Kaggle) exposes a write-through f_locals proxy (PEP 667)
    # and removed PyFrame_LocalsToFast; older versions need the explicit sync.
    proxy = frame.f_locals
    for name, value in values.items(): proxy[name] = value
    if sys.version_info < (3, 13):
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))


def hook_state(hook):
    state = {k: copy.deepcopy(getattr(hook, k)) for k in
             ('rows', 'epochs', 'point_seconds', 'steps', 'tasks', 'batch_counts') if hasattr(hook, k)}
    state['buffer'] = list(hook.buffer.samples)
    state['generators'] = {} if hook.measurement is None else {
        k: g.get_state() for k, g in hook.measurement.generators.items()}
    return state


def restore_hook(hook, state):
    for k, value in state.items():
        if k not in ('buffer', 'generators'): setattr(hook, k, copy.deepcopy(value))
    hook.buffer.samples.clear()
    hook.buffer.samples.extend(state['buffer'])
    if hook.measurement is not None:
        hook.measurement.generators.clear()
        for k, value in state['generators'].items():
            g = dedicated_generator(hook.measurement.seed, hook.measurement.purpose, torch.device(k))
            g.set_state(value.cpu()); hook.measurement.generators[k] = g


class Progress:
    def __init__(self, out, identity, name='latest.pt'):
        self.out, self.identity = Path(out), copy.deepcopy(identity)
        self.path = self.out/name
        self.state = None
        if self.path.exists():
            self.state = torch.load(self.path, map_location='cpu', weights_only=False)
            if self.state.get('identity') != self.identity:
                raise ValueError('resume identity mismatch: ' + str(self.path))

    def save(self, state):
        payload = {**state, 'identity': self.identity}
        payload['artifacts'] = {p.name: p.read_bytes() for p in self.out.iterdir()
                                if p.suffix in ('.csv', '.json') and p.name != 'summary.json'}
        if self.path.exists():
            previous = self.path.with_name(self.path.stem+'.previous.pt')
            temporary = previous.with_suffix('.tmp')
            shutil.copyfile(self.path, temporary); temporary.replace(previous)
        atomic_torch_save(payload, self.path)

    def restore_artifacts(self):
        if self.state:
            for name, data in self.state.get('artifacts', {}).items():
                temporary = self.out/(name+'.tmp'); temporary.write_bytes(data)
                temporary.replace(self.out/name)

    def boundary(self, checkpoint, epoch_seconds=0., point_seconds=0.):
        if deadline_due(epoch_seconds+point_seconds):
            checkpoint()
            atomic_json_save({'status': 'PAUSED', 'identity': self.identity,
                              'resume_supported': True}, self.out/'summary.json')
            log('PAUSED — rerun with this version\'s output attached')
            raise SystemExit(75)


def projected(hook, total_epochs, total_points):
    if len(hook.point_seconds) == 1:
        training = sum(r['seconds'] for r in hook.epochs)/max(1, len(hook.epochs))*total_epochs
        remaining = max(0, total_points-len(hook.point_seconds))
        log(f'projected total seconds={training + hook.point_seconds[0]["seconds"]*remaining:.1f} '
            f'training={training:.1f} remaining_points={remaining}')


class Status:
    """Publish liveness separately from the last actual progress update."""
    def __init__(self, out, job):
        self.path = Path(out)/'status.json'
        self.values = {'job': job, 'task': None, 'epoch': None, 'step': 0,
                       'last_progress_time': time.time()}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.update(phase='starting')
        self.thread.start()

    def update(self, **values):
        with self.lock:
            self.values.update(values, last_progress_time=time.time())
            self._write()

    def _write(self):
        atomic_json_save({**self.values, 'last_update_time': time.time()}, self.path)

    def _heartbeat(self):
        while not self.stop.wait(30):
            with self.lock: self._write()

    def close(self):
        self.stop.set(); self.thread.join(timeout=2)


def deadline_due(seconds=0.):
    deadline = os.environ.get('VNA_DEADLINE')
    return bool(deadline and float(deadline)-time.time() < seconds+600)


def pause_if_due(checkpoint=lambda: None, seconds=0.):
    if deadline_due(seconds):
        checkpoint()
        log("PAUSED — rerun with this version's output attached")
        raise SystemExit(75)


def print_heartbeat(root, started, deadline, lanes):
    """Shared notebook heartbeat: logs, runner status, and GPU utilization."""
    import json
    import subprocess
    statuses = {}
    for path in [*Path(root).rglob('status.json'), *Path(root).rglob('heartbeat.json')]:
        try: statuses[str(path.relative_to(root))] = json.loads(path.read_text())
        except (OSError, ValueError): pass
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                          '--format=csv,noheader'], capture_output=True, text=True)
    print(f'heartbeat elapsed={time.time()-started:.0f}s deadline_in={deadline-time.time():.0f}s '
          f'lanes={lanes} status={statuses} gpu={gpu.stdout.strip()}', flush=True)


def launch_logged(cmd, *, cwd, env, log_path, lane, job, lanes, paused, deadline):
    """Stream notebook output and propagate the resumable exit code."""
    import subprocess
    if paused.is_set() or time.time() >= deadline-600:
        paused.set()
        return False
    lanes[lane] = {'job': job, 'last': 'starting'}
    env = {**env, 'VNA_LANE': str(lane), 'VNA_JOB': job}
    with Path(log_path).open('a', buffering=1) as stream:
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            stream.write(line)
            lanes[lane]['last'] = line.rstrip()
            print(line.rstrip() if line.startswith(f'[{lane} {job}]') else
                  f'[{lane} {job}] {line.rstrip()}', flush=True)
        code = process.wait()
    lanes[lane]['job'] = 'idle'
    if code == 75:
        paused.set()
        return False
    if code: raise subprocess.CalledProcessError(code, cmd)
    return True


# Measurement instrumentation shares Status's daemon/writer and Progress's
# boundary checkpoints; it never interrupts or replaces the training deadline.
from contextvars import ContextVar
import csv
import faulthandler
import json

current_measurement = ContextVar('vna_measurement', default=None)


class MeasurementTimeout(TimeoutError):
    pass


def measurement_update(**values):
    monitor = current_measurement.get()
    if monitor:
        monitor.check_deadline()
        monitor.update(**values)


def measurement_cli(parser):
    parser.add_argument('--measurement-timeout-s', type=float, default=7200.)
    parser.add_argument('--faulthandler-timeout-s', type=float, default=3600.)


def configure_measurement(measure, args, out):
    if measure:
        measure.runtime = {'out': str(out),
                           'timeout_s': getattr(args, 'measurement_timeout_s', 7200.),
                           'traceback_s': getattr(args, 'faulthandler_timeout_s', 3600.)}
    return measure


class MeasurementMonitor(Status):
    def __init__(self, metadata, ctx, *, out=None, timeout_s=7200., traceback_s=3600.):
        self.started = time.monotonic()
        self.timeout_s, self.traceback_s = timeout_s, traceback_s
        self.out = Path(out) if out else None
        if self.out: self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out/'heartbeat.json' if self.out else None
        self.values = {**metadata, **{k: ctx.get(k) for k in ('task', 'epoch', 'step', 'point')},
                       'mu': None, 'mu_rel': None, 'site': None, 'rank': None, 'solve': None, 'cg_iteration': 0}
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.solves = []
        self.traceback_file = None

    def _write(self):
        self.values['point_elapsed_seconds'] = time.monotonic()-self.started
        if self.path: atomic_json_save({**self.values, 'last_update_time': time.time()}, self.path)

    def _heartbeat(self):
        while not self.stop.wait(60):
            with self.lock: self._write()

    def check_deadline(self):
        if time.monotonic()-self.started > self.timeout_s:
            raise MeasurementTimeout('measurement soft deadline exceeded')

    def begin_solve(self):
        self.check_deadline()
        self.update(cg_iteration=0, solve_index=len(self.solves)+1)

    def iteration(self, iteration):
        self.check_deadline()
        with self.lock: self.values['cg_iteration'] = iteration

    def end_solve(self, record):
        keys = ('setting', 'benchmark', 'seed', 'task', 'epoch', 'step', 'point',
                'mu', 'mu_rel', 'site', 'rank', 'solve', 'solve_index')
        row = {**{k: self.values.get(k) for k in keys},
               'seconds': record['seconds'], 'cg_iters_used': record['iterations'],
               'converged': record['converged']}
        self.solves.append(row)
        if self.out:
            path = self.out/'solves.csv'
            with path.open('a', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if stream.tell() == 0: writer.writeheader()
                writer.writerow(row)

    def __enter__(self):
        self.token = current_measurement.set(self)
        self.update(phase='measurement')
        self.thread.start()
        if self.out and self.traceback_s > 0:
            self.traceback_file = (self.out/'measurement_tracebacks.log').open('a')
            faulthandler.dump_traceback_later(self.traceback_s, repeat=True, file=self.traceback_file)
        return self

    def __exit__(self, *exc):
        if self.traceback_file:
            faulthandler.cancel_dump_traceback_later()
            self.traceback_file.close()
        self.close()
        with self.lock: self._write()
        current_measurement.reset(self.token)
        seconds = sum(r['seconds'] for r in self.solves)
        log(f'measurement CG solves={len(self.solves)} mean_solve_seconds={seconds/max(1, len(self.solves)):.6f} '
            f'point_seconds={time.monotonic()-self.started:.6f}')

    def timeout_rows(self, measure, ctx):
        snapshot = json.dumps(self.values)
        return [{**measure.metadata,
                 **{k: ctx[k] for k in ('task', 'epoch', 'step', 'point', 'end_of_task', 'train_acc') if k in ctx},
                 'mu_rel': mu, 'rank': rank, 'damping_norm': 'outputs', 'status': 'timeout',
                 'cg_converged_real': False, 'cg_converged_null': False, 'cg_converged': False,
                 'last_heartbeat': snapshot, 'wall_seconds': time.monotonic()-self.started}
                for mu in measure.mus for rank in measure.ranks]
