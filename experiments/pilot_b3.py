"""Six performance arms through pinned PyCIL and the existing resume hooks."""
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import argparse
import sys
import torch
from experiments import stage_b3
from experiments.pilot_gir_vna import PilotEngine, choose_interval
from adapters.pilot_icarl import ICaRLPilot
from diagnostics.icarl_measure import B3Measurement

ARMS = ('icarl','icarl_vna','icarl_ngd','icarl_gir','icarl_gir_randdir','icarl_gir_vna')


def run(args):
    args.init_seed = 0
    every_epoch = False
    every = args.vna_every
    if args.schedule:
        schedule = json.loads(Path(args.schedule).read_text())
        if not args.fake_data and not schedule.get('authoritative', False):
            raise ValueError('real full runs require the shared icarl_vna calibration schedule')
        if schedule.get('synthetic') and not args.fake_data:
            raise ValueError('synthetic CPU calibration cannot authorize a real full run')
        every = schedule['K']
        every_epoch = schedule.get('every_epoch', False)
        if not isinstance(every, int) or every < 1: raise ValueError('schedule has no feasible K')
    if not args.smoke and ('vna' in args.arm or 'ngd' in args.arm) and args.schedule is None and not args.auto_schedule:
        raise ValueError('full corrections require a preregistered --schedule from two-task smoke')
    engine = PilotEngine(seed=args.seed, gir='gir' in args.arm, random='randdir' in args.arm,
        coefficient=args.lambda_vna if ('vna' in args.arm or 'ngd' in args.arm or args.calibrate) else 0.,
        ngd='ngd' in args.arm, every=every, rank=4, pool=args.gir_pool,
        refresh=args.gir_refresh_steps, rho_min=args.rho_min, selection=args.selection)
    intervention = ICaRLPilot(engine, window=4 if args.fake_data else 128,
        calibrate=args.calibrate, every_epoch=every_epoch, auto_schedule=args.auto_schedule)
    # Only the upstream control collects quick diagnostics. Empty callbacks keep
    # all arms on the same lifecycle, without invoking Stage-B matrices.
    measure = B3Measurement(dict(setting='B3', seed=args.seed, smoke=args.smoke,
        fake_data=args.fake_data, arm=args.arm), measure_profile='quick')
    if args.arm != 'icarl' or args.no_measure:
        class NoMeasurement(B3Measurement):
            def __call__(self, *a, **kw): return []
        measure = NoMeasurement(measure.metadata, measure_profile='quick')
    args.measure_profile = 'quick'
    args.save_measurement_checkpoints = args.arm == 'icarl' and not args.no_measure
    # PyCIL repeats its final epoch description through logging.info; the hook
    # owns the pilot epoch lines, while measurement and notebook heartbeats stay live.
    previous_logging = logging.root.manager.disable
    try:
        logging.disable(logging.INFO)
        hook = stage_b3.run(args, measure_override=measure, intervention=intervention)
    finally:
        logging.disable(previous_logging)
    out = Path(args.out)
    if args.calibrate:
        points = intervention.calibration
        if len(points)!=2: raise ValueError('calibration must include task 1 and task 2')
        for point in points: point['iteration_seconds'] = float(__import__('numpy').median(intervention.iteration_times))
        schedule = choose_interval(points, min(intervention.full_batches))
        schedule['synthetic'] = args.fake_data
        schedule['arm'] = args.arm
        schedule['authoritative'] = args.arm == 'icarl_vna'
        schedule['code_commit'] = __import__('experiments.vna_provenance', fromlist=['code_commit']).code_commit()
        stage_b3.write_json(out/'schedule.json', schedule)
        if not args.fake_data and schedule['authoritative']:
            path = stage_b3.ROOT/'docs/vna/PREREGISTRATION_CHANGES.md'
            with path.open('a') as f:
                f.write('\nB3 K selected from real smoke before full training: `'+json.dumps(schedule, sort_keys=True)+'`.\n')
            (out/'PREREGISTRATION_CHANGES.md').write_text(path.read_text())
        print('PREREGISTERED K', json.dumps(schedule), flush=True)
    if hook.point_seconds:
        # quick uses 64 eval samples: final measurement n_out is 6,400.
        estimate = max(p['seconds']*6400/(p['batch_size']//2*(10*p['task'])) for p in hook.point_seconds)*3
        print('quick projected total measurement seconds (3 points, linear n_out to 6400):', estimate, flush=True)
    return hook


def parser():
    p = stage_b3.parser()
    p.set_defaults(out='outputs/pilot_b3', measure_profile='quick')
    p.add_argument('--arm', choices=ARMS, default='icarl')
    p.add_argument('--lambda-vna', type=float, default=1.)
    p.add_argument('--vna-every', type=int, default=100)
    p.add_argument('--schedule')
    p.add_argument('--auto-schedule', action='store_true')
    p.add_argument('--selection', choices=('tiny_V','tiny_V_euclid','random_frob'), default='tiny_V')
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--no-measure', action='store_true')
    p.add_argument('--gir-pool', type=int, default=8)
    p.add_argument('--gir-refresh-steps', type=int, default=1000)
    p.add_argument('--rho-min', type=float, default=0.)
    return p


def remeasure(args):
    """Re-run quick diagnostics from immutable mid-task measurement states."""
    source = Path(args.from_dir).resolve()
    checkpoints = sorted((source/'measurement_checkpoints').glob('task_*_epoch_*.pt'))
    if not checkpoints:
        raise FileNotFoundError('no task 2/5/10 measurement checkpoints in '+str(source))
    output = Path(args.out or source/'remeasured_quick').resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    wanted = {2, 5, 10}
    first = min(int(torch.load(p, map_location='cpu', weights_only=False)['task']) for p in checkpoints)
    for path in checkpoints:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['task'] not in wanted: continue
        learner = SimpleNamespace(**payload['learner'])
        learner._network.to(args.device)
        if learner._old_network is not None: learner._old_network.to(args.device)
        batch = tuple(t.to(args.device) for t in payload['buffer'])
        metadata = dict(setting='B3', seed=args.seed, arm='icarl', smoke=False,
                        fake_data=False, measure_profile='quick', remeasured=True,
                        checkpoint=str(path))
        measure = B3Measurement(metadata, measure_profile='quick')
        measure.runtime = {'out': str(output), 'timeout_s': args.measurement_timeout_s,
                           'traceback_s': args.faulthandler_timeout_s}
        from adapters.icarl_hooks import Prediction, loss_context
        model = Prediction(learner._network)
        spec, loss_ctx = loss_context(learner, batch)
        ctx = {**loss_ctx, 'learner': learner, 'task': payload['task'],
               'epoch': payload['epoch'], 'step': payload['step'],
               'end_of_task': payload['epoch'] == payload['last'],
               'train_acc': payload['train_acc'],
               'first_measurement_point': payload['task'] == first}
        generator = torch.Generator(device=torch.device(args.device)).manual_seed(args.seed)
        rows.extend(measure(model, batch, spec, ctx, generator=generator))
    if not rows:
        raise FileNotFoundError('measurement checkpoints do not contain tasks 2, 5, or 10')
    stage_b3.write_csv(output/'measurements.csv', rows)
    from diagnostics.vna_gate import evaluate_gate, setting_config
    reproduction_path = source/'reproduction_gate.json'
    reproduction = json.loads(reproduction_path.read_text()) if reproduction_path.exists() else {'status': 'UNKNOWN'}
    cfg = setting_config('B3', mu_rel_default=.01); cfg['min_points'] = 0
    gate = evaluate_gate(rows, cfg)
    gate.update(reproduction=reproduction, measurements_valid=reproduction.get('status') == 'PASS')
    if not gate['measurements_valid']: gate['status'] = 'INVALID'
    result = {'by_mu': {'0.01': gate}, 'official_mu': .01,
              'gate_null_statistic': 'nu', 'status': gate['status'],
              'status_label': 'exploratory', 'remeasured_from': str(source)}
    stage_b3.write_json(output/'gate_quick.json', result)
    return result


def remeasure_parser():
    p = argparse.ArgumentParser(description='Remeasure saved iCaRL quick checkpoints')
    p.add_argument('--from', dest='from_dir', required=True)
    p.add_argument('--out')
    p.add_argument('--device', choices=('cpu','cuda:0'), default='cuda:0')
    p.add_argument('--seed', type=int, default=1993)
    p.add_argument('--measurement-timeout-s', type=float, default=7200.)
    p.add_argument('--faulthandler-timeout-s', type=float, default=3600.)
    return p


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'remeasure':
        return remeasure(remeasure_parser().parse_args(sys.argv[2:]))
    return run(parser().parse_args())

if __name__ == '__main__': main()
