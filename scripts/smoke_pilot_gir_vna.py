"""Synthetic CPU wiring/timing smokes only; never runs real MNIST/CIFAR pilots."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',choices=('b5','b3','all'),default='all')
    p.add_argument('--out',default='/tmp/vna_pilot_cpu')
    args = p.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    environment = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
                   'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION':'python', 'REQUIRE_UPSTREAM':'1', 'TQDM_DISABLE':'1'}
    for part in (('b5','b3') if args.part=='all' else (args.part,)):
        module = 'experiments.pilot_'+part
        from experiments.pilot_b5 import ARMS as b5_arms
        from experiments.pilot_b3 import ARMS as b3_arms
        for arm in b5_arms if part=='b5' else b3_arms:
            destination = out/part/arm
            command = [sys.executable,'-u','-m',module,'--arm',arm,'--smoke','--fake-data','--out',str(destination)]
            if part=='b5':
                command += ['--seed','0']
                if arm.startswith('cbp'): command += ['--replacement-rate','0.001']
            else:
                command += ['--device','cpu','--workers','0']
                if 'vna' in arm or 'ngd' in arm: command += ['--calibrate']
            with (out/f'{part}_{arm}.log').open('w') as stream:
                subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
            paths = list(destination.glob('*/seed*/summary.json')) if part=='b5' else [destination/'summary.json']
            for path in paths:
                summary=json.loads(path.read_text())
                print(part,arm,json.dumps({k:summary[k] for k in (
                    ('seconds','seconds_per_step','estimated_full_seconds','correction_seconds','gir_refresh_seconds')
                    if part=='b5' else ('seconds','training_seconds','calibration','measurement_point_seconds'))}),flush=True)

if __name__=='__main__': main()
