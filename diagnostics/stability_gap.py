"""Stability-gap replay of a registered P1 job through task 5 on its own pinned source.

Run this file by absolute path with --repo pointing at a checkout of the P1 commit. The
native P1 main_cil runs unchanged; old/new-class test accuracy is read inside
MainHook.before_optimizer_step (state after all previous SGD updates and interventions)
under no_grad with every RNG and module mode restored. Measurements are reported only if
epoch losses and tasks.json for tasks 1-5 equal the attached P1 run exactly.
"""
import argparse
from contextlib import contextmanager
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import sys

STEP_EVERY, STEP_WINDOW, MAX_TASKS = 5, 200, 5
VOLATILE = ('second', 'time', 'elapsed', 'wall')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=1)); tmp.replace(path)


def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return default


def cli_from_protocol(main, protocol, args, schedule_path):
    """Rebuild the exact P1 command line from its recorded protocol (only --max-tasks differs)."""
    argv = ['--arm', protocol['arm'], '--seed', str(protocol['seed']), '--backbone', protocol['backbone'],
            '--class-order', protocol['class_order_kind'], '--nobn-lr-scale', str(protocol['nobn_lr_scale']),
            '--coefficient', str(protocol.get('coefficient', 1.)), '--max-tasks', str(MAX_TASKS),
            '--device', args.device, '--data-dir', args.data_dir, '--out', str(args.out/'run')]
    options = {a.dest for a in main.parser()._actions}
    for key in ('curvature', 'solver'):
        if key in options and protocol.get(key) is not None: argv += ['--'+key, str(protocol[key])]
    if 'gir_live' in options and protocol.get('gir_live'): argv.append('--gir-live')
    if protocol.get('schedule'): argv += ['--schedule', str(schedule_path)]
    return main.parser().parse_args(argv)


class Tracker:
    def __init__(self, out, torch, np):
        self.torch, self.np = torch, np
        self.path = out/'stability_gap'/'measurements.json'
        self.state = read(self.path, dict(tasks={}))  # survives an exit-75 resume
        self.out, self.data = out, None

    @contextmanager
    def isolated(self, net):
        torch, np = self.torch, self.np
        states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        modes = {m: m.training for m in net.modules()}
        try:
            net.eval()
            with torch.no_grad(): yield
        finally:
            for m, mode in modes.items(): m.training = mode
            random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
            if states[3] is not None: torch.cuda.set_rng_state_all(states[3])

    def test_tensors(self, learner):
        torch = self.torch
        dataset = learner.test_loader.dataset  # test transforms are deterministic
        items = [dataset[i] for i in range(len(dataset))]
        return torch.stack([x for _, x, _ in items]), torch.tensor([int(y) for _, _, y in items])

    def accuracy(self, learner):
        torch = self.torch
        net, known = learner._network, learner._known_classes
        x, y = self.data
        predictions = []
        for start in range(0, len(x), 500):
            logits = net(x[start:start+500].to(learner._device))['logits']
            predictions.append(logits.argmax(1).cpu())
        correct = (torch.cat(predictions) == y).double()
        old, new = y < known, y >= known
        return dict(old=float(100*correct[old].mean()), new=float(100*correct[new].mean()),
                    all=float(100*correct.mean()))

    def begin(self, hook, learner, task):
        key = str(task)
        with self.isolated(learner._network): self.data = self.test_tensors(learner)
        if key not in self.state['tasks']:
            previous = hook.tasks[-1] if hook.tasks else {}
            self.state['tasks'][key] = dict(task=task, offset=hook.steps-1, known=learner._known_classes,
                total=learner._total_classes, batches_per_epoch=len(learner.train_loader),
                pre_switch_old_cnn=previous.get('cnn_top1'), pre_switch_old_nme=previous.get('nme_top1'),
                points=[])
            # Task boundary: latest.pt is the start-of-task checkpoint, latest.previous.pt the end of task-1.
            target = self.out/'stability_gap'/'checkpoints'/f'end_task{task-1}'
            target.mkdir(parents=True, exist_ok=True)
            for name in ('latest.pt', 'latest.previous.pt'):
                source = self.out/'run'/name
                if source.exists(): shutil.copy2(source, target/name)
        self.task = task

    def measure(self, hook, learner, r, kind):
        entry = self.state['tasks'][str(self.task)]
        if any(p['r'] == r and p['kind'] == kind for p in entry['points']): return
        with self.isolated(learner._network): value = self.accuracy(learner)
        entry['points'].append(dict(r=r, kind=kind, global_step=hook.steps-1 if kind != 'task_end' else hook.steps,
                                    epoch=getattr(hook, 'current_epoch', None), **value))
        write(self.path, self.state)

    def before_step(self, hook):
        learner = hook.learner
        task = learner._cur_task+1
        if not 2 <= task <= MAX_TASKS: return
        if self.data is None or getattr(self, 'task', None) != task: self.begin(hook, learner, task)
        r = hook.steps-1-self.state['tasks'][str(task)]['offset']  # SGD updates already applied in this task
        if (r <= STEP_WINDOW and r % STEP_EVERY == 0) or (r > STEP_WINDOW and getattr(hook, 'epoch_step', 0) == 1):
            self.measure(hook, learner, r, 'step' if r <= STEP_WINDOW else 'epoch')

    def task_end(self, hook, learner):
        task = learner._cur_task+1
        if not 2 <= task <= MAX_TASKS or getattr(self, 'task', None) != task: return
        r = hook.steps-self.state['tasks'][str(task)]['offset']
        self.measure(hook, learner, r, 'task_end')


def compare(reference, replay):
    """Exact equality of epoch losses and task rows for tasks 1-5 (timing fields excluded)."""
    diffs = []
    def keyed(rows, key): return {tuple(r[k] for k in key): r for r in rows if r.get('task', 0) <= MAX_TASKS}
    ref_e, new_e = keyed(reference['epochs'], ('task', 'epoch')), keyed(replay['epochs'], ('task', 'epoch'))
    if set(ref_e) != set(new_e): diffs.append(dict(kind='epochs', missing=sorted(set(ref_e)-set(new_e))[:5], extra=sorted(set(new_e)-set(ref_e))[:5]))
    for k in sorted(set(ref_e) & set(new_e)):
        for field in ('train_loss', 'train_acc'):
            if ref_e[k].get(field) != new_e[k].get(field):
                diffs.append(dict(kind='epoch', task=k[0], epoch=k[1], field=field, reference=ref_e[k].get(field), replay=new_e[k].get(field)))
    ref_t, new_t = keyed(reference['tasks'], ('task',)), keyed(replay['tasks'], ('task',))
    if set(ref_t) != set(new_t): diffs.append(dict(kind='tasks', reference=sorted(ref_t), replay=sorted(new_t)))
    schema = []
    for k in sorted(set(ref_t) & set(new_t)):
        a, b = ref_t[k], new_t[k]
        if set(a) != set(b): schema.append(dict(task=k[0], only_reference=sorted(set(a)-set(b)), only_replay=sorted(set(b)-set(a))))
        for field in sorted(set(a) & set(b)):
            if any(v in field for v in VOLATILE): continue
            if a[field] != b[field]: diffs.append(dict(kind='task', task=k[0], field=field, reference=a[field], replay=b[field]))
    expected = 100+50*(MAX_TASKS-1)
    if len(new_e) != expected: diffs.append(dict(kind='epoch_count', replay=len(new_e), expected=expected))
    return dict(status='PASS' if not diffs and not schema else 'FAIL', diffs=diffs[:50], diff_count=len(diffs),
                schema=schema, epochs_compared=len(set(ref_e) & set(new_e)), tasks_compared=len(set(ref_t) & set(new_t)))


def summarize(state, events):
    report = []
    for key, entry in sorted(state['tasks'].items(), key=lambda kv: int(kv[0])):
        points = sorted(entry['points'], key=lambda p: (p['r'], p['kind'] == 'task_end'))
        pre = entry['pre_switch_old_cnn']
        low = min(points, key=lambda p: p['old']) if points else None
        after = [p for p in points if low and p['r'] >= low['r']]
        recover = next((p['r'] for p in after if pre is not None and p['old'] >= pre), None)
        half = next((p['r'] for p in after if pre is not None and p['old'] >= low['old']+.5*(pre-low['old'])), None)
        task_events = [e for e in events if e.get('task') == entry['task']]
        report.append(dict(task=entry['task'], pre_switch_old=pre, r0_old=points[0]['old'] if points else None,
            min_old=low and low['old'], min_old_step=low and low['r'], gap_depth=None if pre is None or low is None else pre-low['old'],
            recovery_step_to_pre_switch=recover, recovery_step_to_half_gap=half, end_old=points[-1]['old'] if points else None,
            end_new=points[-1]['new'] if points else None, curve=points,
            interventions=[dict(step=e['iteration']-entry['offset'], L=e.get('L_before'), KD=e.get('KD_before'),
                V_O=e.get('V_O'), h=e.get('h_dot_d'), g_dot_d=e.get('g_dot_d'), s=e.get('s'), swaps=e.get('swaps'),
                skipped=e.get('skipped'), skip_reason=e.get('skip_reason'), beta=e.get('beta')) for e in task_events]))
    return report


def run(args):
    repo = Path(args.repo).resolve()
    tool_tree = Path(__file__).resolve().parents[1]
    # Only the P1 checkout may provide project modules (site-packages may live under a local .venv).
    sys.path[:] = [str(repo)]+[p for p in sys.path if p and ('site-packages' in p or not Path(p).resolve().is_relative_to(tool_tree))]
    import numpy as np
    import torch
    from experiments import main_cil as main
    for module in (main, inspect.getmodule(main.MainHook)):
        if not Path(module.__file__).resolve().is_relative_to(repo): raise RuntimeError(f'{module.__name__} not from {repo}')
    reference_dir = Path(args.reference)
    protocol = read(reference_dir/'protocol.json')
    reference = dict(epochs=read(reference_dir/'epochs.json', []), tasks=read(reference_dir/'tasks.json', []))
    if len([r for r in reference['tasks'] if r['task'] <= MAX_TASKS]) < MAX_TASKS:
        raise ValueError(f'reference {reference_dir} has fewer than {MAX_TASKS} tasks')
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True); args.out = out
    if main.code_commit() != (protocol.get('schedule') or {}).get('code_commit', main.code_commit()):
        raise ValueError(f"checkout {main.code_commit()} differs from the P1 schedule commit {protocol['schedule']['code_commit']}")
    schedule_path = out/'schedule.json'
    if protocol.get('schedule'): write(schedule_path, protocol['schedule'])
    tracker = Tracker(out, torch, np)
    original_before, original_record = main.MainHook.before_optimizer_step, main.MainHook.record_task
    def before(self, frame):
        tracker.before_step(self)
        return original_before(self, frame)
    def record(self, learner, result):
        tracker.task_end(self, learner)
        return original_record(self, learner, result)
    main.MainHook.before_optimizer_step, main.MainHook.record_task = before, record
    summary = main.run(cli_from_protocol(main, protocol, args, schedule_path))
    replay = dict(epochs=read(out/'run'/'epochs.json', []), tasks=read(out/'run'/'tasks.json', []))
    check = compare(reference, replay)
    identity = dict(job=args.job, reference=str(reference_dir), reference_code_commit=protocol.get('code_commit'),
                    replay_code_commit=main.code_commit(), K0=protocol.get('K0'), status=summary.get('status'))
    write(out/'stability_gap'/'match.json', dict(identity, **check))
    print(f"P1 MATCH {args.job}: {check['status']} (epochs {check['epochs_compared']}, tasks {check['tasks_compared']}, diffs {check['diff_count']})", flush=True)
    if check['status'] != 'PASS':
        write(out/'stability_gap'/'report.json', dict(identity, status='MISMATCH_NO_REPORT', match=check))
        raise SystemExit(f'P1 replay mismatch for {args.job}; measurements withheld (see match.json)')
    events = read(out/'run'/'interventions.json', [])
    write(out/'stability_gap'/'report.json', dict(identity, status='OK', tasks=summarize(tracker.state, events)))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, help='checkout of the P1 commit')
    p.add_argument('--reference', required=True, help='attached P1 job directory (protocol/epochs/tasks)')
    p.add_argument('--job', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    return p


if __name__ == '__main__': run(parser().parse_args())
