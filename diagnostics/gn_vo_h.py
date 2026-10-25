"""V_O, h and the stability gap on the GroupNorm ResNet-32 (checks the story is not noBN-only).

Baseline iCaRL (GroupNorm, superclass, native recipe) trains tasks 1-5. In tasks 2-5, at
r = 0..10 and every 5 steps to 200 (r = SGD updates since the task switch), old/new-class test
accuracy is read exactly as in stability_gap.py, and a fresh engine (Gram/power10, policy of
icarl_full, VNA step only) applies one full T26 application to a deep COPY of the network and
optimizer on the current training window: V_O, h, L/KD before and after, s, Delta and the T26(f)
would_accept decision are logged and the copy is discarded. The real network, optimizer, RNG and
module modes are checked bitwise unchanged after each point.
"""
import argparse
import copy
import json
from pathlib import Path
import random

from diagnostics.stability_gap import Tracker

DENSE, STEP_EVERY, STEP_WINDOW, MAX_TASKS = 10, 5, 200, 5
FIELDS = ('V', 'V_O', 'h_dot_d', 'g_dot_d', 'beta', 'beta_branch', 'leak', 'epsilon_star', 'chi2',
          'skipped', 'skip_reason', 'rho_tr', 'mu', 'Delta', 's', 's_F_star', 'rejected',
          'L_before', 'L_after', 'KD_before', 'KD_after')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=1)); tmp.replace(path)


def run(args):
    import numpy as np
    import torch
    from experiments import main_cil as main
    from adapters.icarl_hooks import Prediction, sites_for, loss_context
    from methods.vna_v3 import V3Engine, POLICY_VERSION
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    reader = Tracker(out, torch, np)  # same isolated old/new accuracy reader as the P1 replay
    rows, offset = [], {}
    original_before = main.MainHook.before_optimizer_step

    def snapshot(learner, optimizer):
        return (copy.deepcopy(learner._network.state_dict()), copy.deepcopy(optimizer.state_dict()),
                random.getstate(), np.random.get_state(), torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                {m: m.training for m in learner._network.modules()})

    def same(a, b):
        def equal(x, y):
            if isinstance(x, torch.Tensor): return torch.equal(x, y)
            if isinstance(x, dict): return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
            if isinstance(x, (list, tuple)): return len(x) == len(y) and all(equal(p, q) for p, q in zip(x, y))
            if isinstance(x, np.ndarray): return np.array_equal(x, y)
            return x == y
        return all(equal(x, y) for x, y in zip(a, b))

    def before(self, frame):
        learner = self.learner
        task = learner._cur_task+1
        if 2 <= task <= MAX_TASKS:
            if task not in offset:
                offset[task] = self.steps-1
                with reader.isolated(learner._network): reader.data = reader.test_tensors(learner)
            r = self.steps-1-offset[task]
            if r <= DENSE or (r <= STEP_WINDOW and r % STEP_EVERY == 0):
                before_state = snapshot(learner, self.optimizer)
                with reader.isolated(learner._network): accuracy = reader.accuracy(learner)
                batch = tuple(t.to(learner._device) for t in self.current_batch)
                spec, ctx = loss_context(learner, batch)  # teacher logits come from the real old network
                ctx.update(classes=learner._total_classes, task=task)
                # The full application (step + T26f exact check) acts on a discarded copy only.
                network = copy.deepcopy(learner._network)
                optimizer = torch.optim.SGD(network.parameters(), lr=self.optimizer.param_groups[0]['lr'])
                optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
                model = Prediction(network)
                engine = V3Engine('vna', seed=args.seed, coefficient=1., K0=1, curvature='power10', solver='gram')
                engine.begin_task(task)
                event = engine.apply(model, sites_for(model), batch, spec, ctx, optimizer, self.steps,
                                     force=True, apply=True) or {}
                # A tried step either survives T26(f) (s > 0) or is rejected (engine then logs s = 0).
                applied = bool(event.get('rejected')) or (event.get('s') or 0) > 0
                would_accept = applied and not event.get('rejected')
                del network, optimizer, model, engine
                unchanged = same(before_state, snapshot(learner, self.optimizer))
                rows.append(dict(task=task, step=r, read_only_unchanged=unchanged, would_accept=would_accept,
                                 step_applied_to_copy=applied, old_acc=accuracy['old'], new_acc=accuracy['new'],
                                 all_acc=accuracy['all'], **{k: event.get(k) for k in FIELDS}))
                write(out/'gn_vo_h.json', dict(seed=args.seed, backbone='resnet32_gn', groups=8,
                    policy_version=POLICY_VERSION, rows=rows))
                if not unchanged: raise RuntimeError('copy application changed the real training state')
        return original_before(self, frame)

    main.MainHook.before_optimizer_step = before
    argv = ['--arm', 'icarl', '--backbone', 'resnet32_gn', '--class-order', 'superclass', '--seed', str(args.seed),
            '--nobn-lr-scale', '0.5', '--max-tasks', str(MAX_TASKS), '--device', args.device, '--data-dir', args.data_dir, '--out', str(out/'run')]
    if args.fake_data: argv.append('--fake-data')  # CPU smoke check only
    main.run(main.parser().parse_args(argv))
    tasks = json.loads((out/'run'/'tasks.json').read_text())
    final = {t['task']: t for t in tasks}
    gaps = []
    for task in sorted({r['task'] for r in rows}):
        points = [r for r in rows if r['task'] == task]
        pre = final.get(task-1, {}).get('cnn_top1')
        low = min(points, key=lambda r: r['old_acc'])
        gaps.append(dict(task=task, pre_switch_old=pre, r0_old=points[0]['old_acc'], min_old=low['old_acc'],
                         min_old_step=low['step'], gap_depth=None if pre is None else pre-low['old_acc'],
                         end_old_step200=points[-1]['old_acc'], end_task_cnn=final.get(task, {}).get('cnn_top1')))
    measured = [r for r in rows if r.get('skip_reason') not in ('cg_z', 'nonfinite')]
    summary = dict(seed=args.seed, status='OK', applications=len(rows), task1_end_cnn_top1=final.get(1, {}).get('cnn_top1'),
        task1_end_nme_top1=final.get(1, {}).get('nme_top1'), gap_by_task=gaps,
        V_O_positive=sum((r.get('V_O') or 0) > 0 for r in measured),
        h_nonnegative=sum((r.get('h_dot_d') is not None) and r['h_dot_d'] >= 0 for r in measured),
        h_negative=sum((r.get('h_dot_d') is not None) and r['h_dot_d'] < 0 for r in measured),
        would_accept=sum(r['would_accept'] for r in rows), step_applied=sum(r['step_applied_to_copy'] for r in rows),
        all_read_only=all(r['read_only_unchanged'] for r in rows))
    write(out/'gn_vo_h_summary.json', summary)
    print('GN V_O/h:', json.dumps(summary), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--fake-data', action='store_true', help='CPU smoke check only')
    return p


if __name__ == '__main__': run(parser().parse_args())
