"""Offline chain of successive VNA applications at the first task-2 intervention (no training).

Run by absolute path with --repo = checkout of the job's own commit. The job's start-of-task-2
checkpoint is resumed by its native main_cil (command rebuilt from its protocol), which takes the
first task-2 SGD step; at the first intervention up to 8 VNA applications (registered definition,
VNA step only: GIR is switched off on the copied engine) are applied one after another to a COPY
of network/optimizer/engine on the same window, each fully recomputed. The chain stops when the
GGN lambda_max falls below 2/eta = 40 or a step is rejected or skipped. Nothing is trained further.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

MAX_CHAIN, LAMBDA_STOP = 8, 40.  # 2/eta with eta = 0.05


class Done(Exception): pass


def run(args):
    tool = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('stability_gap_tool', tool/'diagnostics'/'stability_gap.py')
    sg = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg)
    repo = Path(args.repo).resolve()
    sys.path[:] = [str(repo)]+[p for p in sys.path if p and ('site-packages' in p or not Path(p).resolve().is_relative_to(tool))]
    import numpy as np
    import torch
    from experiments import main_cil as main
    from methods import vna_v3
    from methods.vna_ggn import GGNProducts
    if not Path(main.__file__).resolve().is_relative_to(repo): raise RuntimeError('main_cil not from the job checkout')
    job = Path(args.job_dir)
    protocol = json.loads((job/'run'/'protocol.json').read_text())
    out = Path(args.out).resolve(); (out/'run').mkdir(parents=True, exist_ok=True)
    (out/'run'/'latest.pt').write_bytes((job/'stability_gap'/'checkpoints'/'end_task1'/'latest.pt').read_bytes())
    schedule_path = out/'schedule.json'
    if protocol.get('schedule'): schedule_path.write_text(json.dumps(protocol['schedule']))
    reader = sg.Tracker(out, torch, np)

    def ggn_lambda(model, sites, batch, spec_, ctx):
        with vna_v3.training_products(model, sites, vna_v3.seed_proposals(sites), batch[0]) as p:
            op = GGNProducts(p, batch[1], spec_, ctx)
            generator = vna_v3.dedicated_generator(0, 'chain-ggn', batch[0].device)
            return vna_v3.power_lambda_max(op.F, op.size, generator, iterations=30, device=batch[0].device)

    rows = []
    def after_step(self, hook, learner, optimizer):
        if self.task != 2: return
        batch = tuple(t.to(learner._device) for t in hook.current_batch)
        spec_, ctx = main.loss_context(learner, batch)
        ctx.update(classes=learner._total_classes, task=self.task)
        probe = main.fixed_probe(learner, 256)[0]
        network = copy.deepcopy(learner._network)
        opt = torch.optim.SGD(network.parameters(), lr=optimizer.param_groups[0]['lr'])
        opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        model = main.Prediction(network); sites = main.sites_for(model)
        engine = copy.deepcopy(self.engine); engine.arm = 'vna'  # VNA step only; no GIR swaps on the copy
        view = SimpleNamespace(_network=network, _known_classes=learner._known_classes, _device=learner._device,
                               test_loader=learner.test_loader)
        with reader.isolated(network): reader.data = reader.test_tensors(learner)
        def state(k, row, step_norm):
            with reader.isolated(network): acc = reader.accuracy(view)
            applied = (row.get('s') or 0) > 0
            return dict(k=k, iteration=hook.steps, s=row.get('s'), s_times_d=step_norm,
                L_window=row.get('L_after') if applied else row.get('L_before'),
                KD_window=row.get('KD_after') if applied else row.get('KD_before'),
                ggn_lambda_max=ggn_lambda(model, sites, batch, spec_, ctx), V_O=row.get('V_O'), h=row.get('h_dot_d'),
                old_acc=acc['old'], new_acc=acc['new'], skipped=row.get('skipped'), skip_reason=row.get('skip_reason'),
                rejected=row.get('rejected'), mu=row.get('mu'))
        L0, KD0 = engine._evaluate_current(model, batch, spec_, ctx)
        rows.append(state(0, dict(L_before=L0, KD_before=KD0), 0.))
        for k in range(1, MAX_CHAIN+1):
            before = [p.detach().clone() for p in network.parameters()]
            row = engine.apply(model, sites, batch, spec_, ctx, opt, hook.steps, force=True, probe_batch=probe) or {}
            step = float(sum((p.detach()-q).square().sum() for p, q in zip(network.parameters(), before)).sqrt())
            rows.append(state(k, row, step))
            print(json.dumps(rows[-1]), flush=True)
            if rows[-1]['ggn_lambda_max'] < LAMBDA_STOP or row.get('skipped') or row.get('rejected') or not row.get('s'):
                rows[-1]['stop'] = 'lambda<40' if rows[-1]['ggn_lambda_max'] < LAMBDA_STOP else (row.get('skip_reason') or 'rejected' if row.get('rejected') else 'no step')
                break
        raise Done()

    main.Intervention.after_step = after_step
    cli = sg.cli_from_protocol(main, protocol, SimpleNamespace(device='cuda:0', data_dir=args.data_dir, out=out, fake_data=False), schedule_path)
    try: main.run(cli)
    except Done: pass
    if not rows: raise RuntimeError('no task-2 intervention reached')
    result = dict(job=args.name, code_commit=main.code_commit(), backbone=protocol['backbone'], rows=rows)
    (out/'vna_chain.json').write_text(json.dumps(result, indent=1))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--job-dir', required=True, help='job output with run/protocol.json and stability_gap/checkpoints/end_task1')
    p.add_argument('--name', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out', required=True)
    return p


if __name__ == '__main__': run(parser().parse_args())
