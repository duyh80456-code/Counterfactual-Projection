"""Offline A/B of the virtual-channel GroupNorm definition at the first task-2 VNA application.

Run by absolute path with --repo = checkout of the GN pilot's commit. The pilot's start-of-task-2
checkpoint is resumed by the native main_cil (same args, same schedule), which takes the first
task-2 SGD step; at the first intervention (iteration 2001) the native application is replaced by
two applications on deep copies of engine, network and optimizer:
  A: the virtual channel bypasses GroupNorm (registered definition);
  B: the virtual channel is normalized by its own per-sample spatial mean/var (a one-channel GN
     group, cf. T14 for BN), affine gamma = 1, beta = 0 (runtime patch only, never committed).
Nothing is trained further and the pilot's outputs are not modified.
"""
import argparse
import copy
import inspect
import json
from pathlib import Path
import shutil
import sys

BYPASS = "                    continue  # registered definition: the virtual channel bypasses GroupNorm (it joins no group)\n"
OWN_NORM = ("                    axes = tuple(range(2, value.ndim))\n"
            "                    value = (value-value.mean(axes, keepdim=True))/(value.var(axes, unbiased=False, keepdim=True)+module.eps).sqrt()\n")


class Done(Exception): pass


def run(args):
    repo = Path(args.repo).resolve()
    tool_tree = Path(__file__).resolve().parents[1]
    sys.path[:] = [str(repo)]+[p for p in sys.path if p and ('site-packages' in p or not Path(p).resolve().is_relative_to(tool_tree))]
    import torch
    from experiments import main_cil as main
    from methods import vna_v3
    from methods.vna_ggn import GGNProducts
    if not Path(main.__file__).resolve().is_relative_to(repo): raise RuntimeError('main_cil not from the pilot checkout')
    pilot = Path(args.pilot)
    protocol = json.loads((pilot/'run'/'protocol.json').read_text())
    recorded = next(e for e in json.loads((pilot/'run'/'interventions.json').read_text()) if e.get('task') == 2)
    out = Path(args.out).resolve(); (out/'run').mkdir(parents=True, exist_ok=True)
    shutil.copy2(pilot/'stability_gap'/'checkpoints'/'end_task1'/'latest.pt', out/'run'/'latest.pt')

    source = inspect.getsource(vna_v3.training_products)
    if source.count(BYPASS) != 1: raise RuntimeError('GN bypass line not found in the pilot engine')
    def definition(name):
        namespace = vars(vna_v3)
        exec(compile(source if name == 'A' else source.replace(BYPASS, OWN_NORM), vna_v3.__file__, 'exec'), namespace)

    def ggn_lambda(model, sites, batch, spec, ctx):
        with vna_v3.training_products(model, sites, vna_v3.seed_proposals(sites), batch[0]) as p:
            op = GGNProducts(p, batch[1], spec, ctx)
            generator = vna_v3.dedicated_generator(0, 'gn-ab-ggn', batch[0].device)
            return vna_v3.power_lambda_max(op.F, op.size, generator, iterations=30, device=batch[0].device)

    results = {}
    def after_step(self, hook, learner, optimizer):
        if self.task != 2: return
        if hook.steps != recorded['iteration']:
            raise RuntimeError(f"first task-2 application at {hook.steps}, pilot recorded {recorded['iteration']}")
        batch = tuple(t.to(learner._device) for t in hook.current_batch)
        spec, ctx = main.loss_context(learner, batch)
        ctx.update(classes=learner._total_classes, task=self.task)
        probe = main.fixed_probe(learner, 256)[0]
        for name in ('A', 'B'):
            definition(name)
            network = copy.deepcopy(learner._network)
            opt = torch.optim.SGD(network.parameters(), lr=optimizer.param_groups[0]['lr'])
            opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            model = main.Prediction(network)
            sites = main.sites_for(model)
            engine = copy.deepcopy(self.engine)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            lam_before = ggn_lambda(model, sites, batch, spec, ctx)
            proposed = {}
            restore = vna_v3.V3Engine._restore_params
            def capture(model_, state):  # the rejected step, read just before T26(f) undoes it
                proposed.update({n: p.detach().clone() for n, p in model_.named_parameters()})
                return restore(model_, state)
            vna_v3.V3Engine._restore_params = staticmethod(capture)
            try: row = engine.apply(model, sites, batch, spec, ctx, opt, hook.steps, force=True, probe_batch=probe) or {}
            finally: vna_v3.V3Engine._restore_params = staticmethod(restore)
            after = proposed or {n: p.detach() for n, p in model.named_parameters()}
            lam_after = ggn_lambda(model, sites, batch, spec, ctx)
            step = {n: float((after[n]-before[n]).norm()) for n in before if float((after[n]-before[n]).norm()) > 0}
            scale = row.get('s') or 0.
            results[name] = dict(lambda_E=row.get('lambda_E'), V_O=row.get('V_O'), h=row.get('h_dot_d'),
                g_dot_d=row.get('g_dot_d'), dFd=row['g_dot_d']/row['s_F_star'] if row.get('s_F_star') else None,
                mu=row.get('mu'), s=scale, s_F_star=row.get('s_F_star'), Delta=row.get('Delta'),
                skipped=row.get('skipped'), skip_reason=row.get('skip_reason'), rejected=row.get('rejected'),
                L_before=row.get('L_before'), L_after=row.get('L_after'), KD_before=row.get('KD_before'), KD_after=row.get('KD_after'),
                ggn_lambda_max_before=lam_before, ggn_lambda_max_after=lam_after,
                step_norm_by_layer=step, d_norm_by_layer={n: v/scale for n, v in step.items()} if scale > 0 else None)
        definition('A')
        keys = ('V_O', 'h_dot_d', 'g_dot_d', 's', 'L_before', 'KD_before', 'mu')
        results['A_matches_pilot'] = {k: results['A'].get({'h_dot_d': 'h'}.get(k, k)) == recorded.get(k) for k in keys}
        raise Done()

    main.Intervention.after_step = after_step
    cli = ['--arm', 'icarl_vna', '--backbone', 'resnet32_gn', '--class-order', protocol['class_order_kind'],
           '--seed', str(protocol['seed']), '--nobn-lr-scale', str(protocol['nobn_lr_scale']), '--solver', protocol['solver'],
           '--curvature', protocol['curvature'], '--max-tasks', str(protocol['expected_tasks']),
           '--schedule', args.schedule, '--device', 'cuda:0', '--data-dir', args.data_dir, '--out', str(out/'run')]
    try: main.run(main.parser().parse_args(cli))
    except Done: pass
    if not results: raise RuntimeError('no task-2 application reached')
    (out/'gn_definition_ab.json').write_text(json.dumps(results, indent=1))
    print('GN DEFINITION A/B', protocol['seed'], json.dumps(results, indent=1), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, help='checkout of the GN pilot commit')
    p.add_argument('--pilot', required=True, help='attached icarl_vna__gn__sX output directory')
    p.add_argument('--schedule', required=True, help='the pilot GN schedule (gn_vna.json)')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out', required=True)
    return p


if __name__ == '__main__': run(parser().parse_args())
