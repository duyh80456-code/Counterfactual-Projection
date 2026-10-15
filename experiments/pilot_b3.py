"""Six performance arms through pinned PyCIL and the existing resume hooks."""
import json
from pathlib import Path
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
    if not args.smoke and ('vna' in args.arm or 'ngd' in args.arm) and args.schedule is None:
        raise ValueError('full corrections require a preregistered --schedule from two-task smoke')
    engine = PilotEngine(seed=0, gir='gir' in args.arm, random='randdir' in args.arm,
        coefficient=args.lambda_vna if ('vna' in args.arm or 'ngd' in args.arm or args.calibrate) else 0.,
        ngd='ngd' in args.arm, every=every, rank=4, pool=args.gir_pool,
        refresh=args.gir_refresh_steps, rho_min=args.rho_min)
    intervention = ICaRLPilot(engine, window=4 if args.fake_data else 128, calibrate=args.calibrate, every_epoch=every_epoch)
    # Only the upstream control collects quick diagnostics. Empty callbacks keep
    # all arms on the same lifecycle, without invoking Stage-B matrices.
    measure = B3Measurement(dict(setting='B3', seed=1993, smoke=args.smoke, fake_data=args.fake_data, arm=args.arm), measure_profile='quick')
    if args.arm != 'icarl' or args.no_measure:
        class NoMeasurement(B3Measurement):
            def __call__(self, *a, **kw): return []
        measure = NoMeasurement(measure.metadata, measure_profile='quick')
    args.measure_profile = 'quick'
    hook = stage_b3.run(args, measure_override=measure, intervention=intervention)
    out = Path(args.out)
    if args.calibrate:
        points = intervention.calibration
        if len(points)!=2: raise ValueError('calibration must include task 1 and task 2')
        for point in points: point['iteration_seconds'] = float(__import__('numpy').median(intervention.iteration_times))
        try:
            schedule = choose_interval(points, min(intervention.full_batches))
        except ValueError:
            if not args.fake_data: raise
            schedule = dict(K=None, status='NO_FEASIBLE_K_SYNTHETIC', smoke_points=points,
                            iterations_per_epoch=min(intervention.full_batches))
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
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--no-measure', action='store_true')
    p.add_argument('--gir-pool', type=int, default=8)
    p.add_argument('--gir-refresh-steps', type=int, default=1000)
    p.add_argument('--rho-min', type=float, default=0.)
    return p

if __name__ == '__main__': run(parser().parse_args())
