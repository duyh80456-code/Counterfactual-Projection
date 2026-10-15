"""One fake-data CPU point per setting; no training, downloads, or GPU estimate."""
import csv
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch
from torch import nn
from adapters.lop_hooks import Prediction
from diagnostics.icarl_measure import B3Measurement
from diagnostics.lop_measure import B5Measurement
from diagnostics.vna_align import MeasurementHook
from experiments.vna_resume import MeasurementMonitor
from methods.vna_losses import CE
from methods.vna_sites import HookSite


class FakeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(16)
        self.stem = nn.Conv2d(3, 4, 1)
        self.a = nn.Conv2d(4, 4, 1)
        self.relu = nn.ReLU()
        self.b = nn.Conv2d(4, 4, 1)
        self.head = nn.Linear(4, 3)
    def forward(self, x):
        x = self.stem(self.pool(x))
        return self.head((x+self.b(self.relu(self.a(x)))).mean((2,3)))


class FakeNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 3))
    def predict(self, x):
        features = self.layers[:2](x)
        return self.layers[2](features), [features]


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='outputs/measure_efficiency_smoke')
    args = p.parse_args()
    out = Path(args.out)
    torch.set_num_threads(1)
    torch.manual_seed(45)
    results = []
    for setting in ('B1', 'B2', 'B3', 'B4', 'B5'):
        meta = dict(seed=45, setting=setting)
        ctx = dict(task=2, epoch=1, step=512, end_of_task=False, train_acc=.5, known_classes=1)
        if setting == 'B5':
            model = Prediction(FakeNative()).double().eval()
            batch = torch.randn(256, 3, dtype=torch.float64), torch.arange(256)%3
            shadow = SimpleNamespace(bounds=[.1])
            ctx.update(point='first', within_task_step=256, buffer_task_purity=1., buffer_m=256,
                       shadow=shadow, cbp_candidates=[('cbp',shadow,[torch.tensor([0])])], init_row_norms=[1.],
                       loss_fn=lambda z,c: nn.functional.cross_entropy(z,c['targets']))
            measure = B5Measurement('regression', meta)
            groups = 0
        else:
            model = FakeResNet().double().eval()
            batch = torch.randn(512, 3, 32, 32, dtype=torch.float64), torch.arange(512)%3
            def sites(m): return [HookSite(m, 'a', 'b', 'conv', between=('relu',))]
            def context(c,b): return CE(), {'targets': b[1]}
            measure = B3Measurement(meta, sites_fn=sites, context_fn=context,
                                    ranks=(4,8,16) if setting == 'B3' else (4,16,64), groups=setting=='B3')
            groups = 2 if setting == 'B3' else 0
        path = out/setting
        path.mkdir(parents=True, exist_ok=True)
        (path/'solves.csv').unlink(missing_ok=True)
        measure.runtime = dict(out=path, traceback_s=0)
        started = time.monotonic()
        rows = MeasurementHook(measure)(model, batch, CE(), ctx)
        elapsed = time.monotonic()-started
        solves = list(csv.DictReader((path/'solves.csv').open()))
        rank_sites = len(rows)//2
        ranks = len({r['rank'] for r in rows})
        # Compare at the same two mu values, eight N1+N2 nulls and valid ranks.
        criteria = sum(rows[0].get(k, 0)>0 for k in ('cbp_m','redo_m')) if setting=='B5' else 0
        before = 2*(55*rank_sites + 2*ranks + (6*rank_sites*criteria if setting=='B5' else 2*ranks*groups))
        expected_after = 2*(34+groups+2*rank_sites+2*criteria)
        assert len(solves) == expected_after, (setting, len(solves), expected_after)
        record = dict(setting=setting, fake_data=True, sites=1, valid_ranks=sorted({r['rank'] for r in rows}),
                      old_cg_solves=before, new_cg_solves=len(solves), point_seconds=elapsed,
                      mean_solve_seconds=sum(float(r['seconds']) for r in solves)/len(solves))
        results.append(record)
        print(json.dumps(record), flush=True)
    (out/'costs.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__': main()
