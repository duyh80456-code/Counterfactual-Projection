"""Bounded measurement smoke; CPU costs are not T4 timing estimates."""
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
    p.add_argument('--batch-size', type=int, default=32, help='Even fake-data batch; real-model memory profile always uses 256')
    p.add_argument('--solver', choices=['auto','output_exact','cg'], default='auto')
    p.add_argument('--output-exact-max', type=int, default=4096)
    p.add_argument('--profile-real-tiny', action='store_true', help='Profile actual CIFAR models, batch 256, requires CUDA')
    p.add_argument('--out', default='outputs/measure_efficiency_smoke')
    args = p.parse_args()
    out = Path(args.out)
    torch.set_num_threads(1)
    torch.manual_seed(45)
    results = []
    probe = torch.randn(256,256,dtype=torch.float64)
    started = time.perf_counter()
    torch.linalg.eigh(probe+probe.T)
    cubic_calibration = (time.perf_counter()-started)/256**3
    for setting in ('B1', 'B2', 'B3', 'B4', 'B5'):
        meta = dict(seed=45, setting=setting)
        ctx = dict(task=2, epoch=1, step=512, end_of_task=False, train_acc=.5, known_classes=1)
        if setting == 'B5':
            model = Prediction(FakeNative()).double().eval()
            batch = torch.randn(args.batch_size, 3, dtype=torch.float64), torch.arange(args.batch_size)%3
            shadow = SimpleNamespace(bounds=[.1], bias_corrected_util=[torch.arange(4)])
            ctx.update(point='first', within_task_step=256, buffer_task_purity=1., buffer_m=256,
                       shadow=shadow, cbp_candidates=[('cbp',shadow,[torch.tensor([0])])], init_row_norms=[1.],
                       loss_fn=lambda z,c: nn.functional.cross_entropy(z,c['targets']))
            measure = B5Measurement('regression', meta)
            groups = 0
        else:
            model = FakeResNet().double().eval()
            batch = torch.randn(args.batch_size, 3, 32, 32, dtype=torch.float64), torch.arange(args.batch_size)%3
            def sites(m): return [HookSite(m, 'a', 'b', 'conv', between=('relu',))]
            def context(c,b): return CE(), {'targets': b[1]}
            measure = B3Measurement(meta, sites_fn=sites, context_fn=context,
                                    ranks=(4,8,16) if setting == 'B3' else (4,16,64), groups=setting=='B3')
            groups = 2 if setting == 'B3' else 0
        measure.solver, measure.output_exact_max = args.solver, args.output_exact_max
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

        record = dict(setting=setting, fake_data=True, batch_size=args.batch_size, sites=1, valid_ranks=sorted({r['rank'] for r in rows}),
                      old_cg_solves=before, new_cg_solves=len(solves), point_seconds=elapsed,
                      mean_solve_seconds=sum(float(r['seconds']) for r in solves)/len(solves))
        factors = json.loads((path/'factors.json').read_text())
        n_out = rows[0]['n_out']
        build = sum(r['M_build_seconds'] for r in factors)
        factor = sum(r['factor_seconds'] for r in factors)
        eval_factor = next((r for r in factors if r['n_out']==n_out), None)
        forced_n_out = 25600 if setting != 'B5' else 1280
        record.update(solver=rows[0]['solver'], n_out=n_out, M_build_seconds=build,
                      factor_seconds=factor, M_memory_bytes=max((r['M_memory_bytes'] for r in factors), default=0),
                      projected_run_measurement_cpu_hours=elapsed*dict(B1=15,B2=15,B3=13,B4=12,B5=38)[setting]/3600,
                      projected_gpu_hours=None, gpu_timing_available=False,
                      forced_output_exact_n_out=forced_n_out, forced_output_exact_M_bytes=forced_n_out**2*8,
                      forced_output_exact_factor_cpu_seconds_n_cubed=(eval_factor['factor_seconds']*(forced_n_out/n_out)**3 if eval_factor else cubic_calibration*forced_n_out**3),
                      forced_output_exact_cost_basis='CPU fake-model factor timing; cubic factorization estimate, not a T4 forecast',
                      peak_cuda_bytes=None, cpu_process_peak_rss_bytes=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss * (1 if __import__('sys').platform == 'darwin' else 1024))
        results.append(record)
        print(json.dumps(record), flush=True)
    if args.profile_real_tiny: profile_real_tiny(out)
    (out/'costs.json').write_text(json.dumps(results, indent=2)+'\n')


def profile_real_tiny(out):
    """Actual 256-example eval batch, no training or downloads."""
    if not torch.cuda.is_available():
        rows = [dict(model=name,batch_size=256,tiny_V_peak_cuda_bytes=None,
                     status='unavailable: CUDA is not available')
                for name in ('resnet8x4','wrn_16_2','ResNet32')]
        for row in rows: print(json.dumps(row),flush=True)
        (out/'tiny_memory.json').write_text(json.dumps(rows,indent=2)+'\n')
        return
    from adapters.repdistiller_hooks import upstream, sites_for
    from experiments.stage_b3 import upstream_imports
    from adapters.icarl_hooks import Prediction as ICarlPrediction, sites_for as icarl_sites
    from methods.vna_select import tiny_V
    results = []
    for name in ('resnet8x4','wrn_16_2','ResNet32'):
        if name == 'ResNet32':
            upstream_imports()
            from utils.inc_net import IncrementalNet
            native = IncrementalNet({'convnet_type':'resnet32'}, False)
            native.update_fc(100)
            model = ICarlPrediction(native).cuda().eval(); sites = icarl_sites(model)
        else:
            with upstream() as models: model = models[name](num_classes=100).cuda().eval()
            sites = sites_for(model)
        batch = torch.randn(256,3,32,32,device='cuda'), torch.arange(256,device='cuda')%100
        with torch.no_grad(): z = torch.randn_like(model(batch[0])).double()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        for site in sites: tiny_V(model,site,[batch],min(16,site.first_layer.layer.weight.shape[0]),cotangents=[z])
        torch.cuda.synchronize()
        row = dict(model=name,batch_size=256,tiny_V_peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                   seconds=time.monotonic()-started)
        print(json.dumps(row),flush=True); results.append(row)
        del model,sites,batch,z,site
        torch.cuda.empty_cache()
    (out/'tiny_memory.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__': main()
