"""B5 widths 5/20/100 x four registered arms x three seeds, using native hooks."""
import argparse
from pathlib import Path
from experiments import pilot_v3
from methods.vna_v3 import V3Engine


def jobs(out, schedule_root, *, fake=False, resume=False):
    result = []
    for width in (5,20,100):
        for arm in ('bp','bp_vna','bp_gir','bp_vna_gir'):
            for seed in (0,1,2):
                argv = ['--setting','regression','--arm','b5_'+arm,'--width',str(width),
                        '--seed',str(seed),'--device','cpu','--out',str(Path(out)/f'width{width}'/arm/f'seed{seed}')]
                if arm != 'bp': argv += ['--schedule',str(Path(schedule_root)/f'b5_width{width}.json')]
                if fake: argv += ['--fake-data']
                if resume: argv += ['--resume']
                result.append(pilot_v3.parser().parse_args(argv))
    return result


def run(args):
    # A legacy engine must never silently produce a purported T26-T28 width sweep.
    if not {'mu','nu'} <= V3Engine().state_dict().keys():
        raise RuntimeError('merge Phase 2 T26-T28 first; the checked-out engine has no persistent mu/nu')
    trials = jobs(args.out,args.schedule_root,fake=args.fake_data,resume=args.resume)
    if not args.fake_data:
        for width in (5,20,100):
            if not (Path(args.schedule_root)/f'b5_width{width}.json').is_file():
                raise ValueError(f'missing width-specific public-runner schedule for width {width}')
    return [pilot_v3.run(trial) for trial in trials]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--schedule-root',required=True)
    p.add_argument('--fake-data',action='store_true')
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
