"""Round-2 numerical identities, production regression and protocol checks."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from diagnostics.measure_efficiency import SolveCache, virtual_energy, null_columns
from diagnostics.stage_a import null_residual
from methods.vna_ggn import GGNConfig
from methods.vna_joint import joint_correction
from methods.vna_losses import CE, KDRepDistiller, MSE
from methods.vna_sites import HookSite, hook_tiny
from methods.vna_select import tiny_V


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(48)
    yield
    torch.set_num_threads(old)


def problem():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 3)).double().eval()
    site = HookSite(model, '0', '2', 'linear', between=('1',))
    data = torch.randn(6, 3, dtype=torch.float64), torch.arange(6)%3
    return model, site, data


@pytest.mark.parametrize('spec', [CE(), KDRepDistiller(), MSE()])
def test_output_exact_cg_all_statistics(spec):
    model, site, data = problem()
    if isinstance(spec, MSE): data = data[0], torch.randn(6, 3, dtype=torch.float64)
    context = {'targets': data[1], 'teacher_logits': torch.randn(6, 3, dtype=torch.float64)}
    cfg = GGNConfig(cg_iters=2000, cg_tol=1e-12, project_bn=False)
    factors = {}
    for mu in (.001, .01):
        exact = SolveCache(model, site, data, spec, context, cfg, mu=mu, solver='output_exact', factors=factors)
        cg = SolveCache(model, site, data, spec, context, cfg, mu=mu, solver='cg')
        e = spec.residual(model(data[0]), context)
        for index, residual in enumerate([e]+[null_residual(e, kind, torch.Generator().manual_seed(i),
                residual_is_zero_sum=spec.residual_is_zero_sum) for kind in ('N1','N2') for i in range(1)]):
            z = exact.solve(str(index), residual, data, context)
            reference = cg.solve(str(index), residual, data, context)
            assert float((z-reference).norm()/reference.norm().clamp_min(1e-30)) <= 1e-8
            A, _ = tiny_V(model, site, [data], 2, cotangents=[z], cfg=cfg)
            exact_d, _, _, record = joint_correction(model, [site], {site:A}, data, cfg,
                mu=mu, loss_spec=spec, ctx=context, cotangent=residual, output_factor=exact.factor(data, context))
            cg_d, _, _, other = joint_correction(model, [site], {site:A}, data, cfg,
                mu=mu, loss_spec=spec, ctx=context, cotangent=residual)
            va = torch.cat([v.flatten() for v in exact_d.values()])
            vb = torch.cat([v.flatten() for v in cg_d.values()])
            assert float((va-vb).norm()/vb.norm().clamp_min(1e-30)) <= 1e-8
            for key in ('V','rho'):
                assert record[key] == pytest.approx(other[key], rel=1e-8, abs=1e-25)
            energy = virtual_energy(model, [site], {site:A}, data, z, residual)
            assert energy['V'] == pytest.approx(other['V'], rel=1e-8, abs=1e-25)
        assert exact.factor(data, context).range_ok
    assert len(factors) == 1


def test_range_flag_and_subset_matrix():
    model, site, data = problem()
    ctx = {'targets': data[1]}
    exact = SolveCache(model, site, data, CE(), ctx, GGNConfig(), solver='output_exact')
    factor = exact.factor(data, ctx)
    mask = torch.tensor([True, False, True, False, True, False])
    subset = tuple(v[mask] for v in data); context = {'targets': subset[1]}
    sub = exact.factor(subset, context, parent=factor, mask=mask)
    independent = SolveCache(model, site, subset, CE(), context, GGNConfig(), solver='output_exact').factor(subset, context)
    torch.testing.assert_close(sub.matrix, independent.matrix, rtol=1e-10, atol=1e-14)
    factor.z(torch.ones_like(factor.root[:, :, 0]), .01)
    assert not factor.range_ok and factor.range_error > .9


@pytest.mark.parametrize('conv', [False, True])
@pytest.mark.parametrize('projection', [False, True])
def test_streamed_tiny_dense_oracle(tmp_path, conv, projection):
    path = tmp_path/'old_sites.py'
    path.write_bytes(subprocess.check_output(['git','show','HEAD:methods/vna_sites.py']))
    module_spec = importlib.util.spec_from_file_location('dense_sites', path)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    if conv:
        model = nn.Sequential(nn.Conv2d(2,3,3,padding=1),nn.ReLU(),nn.Conv2d(3,2,3,padding=1)).double()
        data = torch.randn(11,2,4,4,dtype=torch.float64), torch.zeros(11,dtype=torch.long)
        site = HookSite(model,'0','2','conv',between=('1',))
    else: model, site, data = problem()
    z = torch.randn_like(model(data[0]))
    kwargs = dict(cotangents=[z], cfg=GGNConfig(), existing_weight_projection=projection,
                  normalize=True, score_signs=True, numerical_threshold=1e-12)
    expected, old = module.hook_tiny(model,site,[data],2,**kwargs)
    actual, new = hook_tiny(model,site,[data],2,chunk_size=2,**kwargs)
    for key in ('S','N_z','generalized_eigenvalues'):
        assert float((new[key]-old[key]).norm()/old[key].norm()) <= 1e-10
    for key in actual:
        if actual[key] is not None:
            assert float((actual[key]-expected[key]).norm()/expected[key].norm()) <= 1e-10


def test_nu_gate_and_schedules(tmp_path):
    from diagnostics.vna_gate import evaluate_gate, setting_config
    from adapters.lop_hooks import measurement_tasks
    from experiments.stage_b5 import write_json
    values = {'N1': ([0.]*8,[100.]*8,[dict(V=.1,e_norm2=1.,nu=.1)]*8)}
    stats = null_columns(1.,.2,values,nu=.5)
    assert stats['significant_nu_N1'] and not stats['significant_rho_N1']
    rows = [dict(setting='B5',task=55+5*i,step=i,point='first',rank=16,mu_rel=.01,
        cg_converged=True,rho_joint=.4,V_tiny_V=2.,V_random_frob=1.,**stats) for i in range(10)]
    result = evaluate_gate(rows, setting_config('B5',tasks_run=100))
    assert result['status']=='PASS' and result['points_in_window']==10 and result['gate_null_statistic']=='nu'
    assert evaluate_gate(rows[:-1],setting_config('B5',tasks_run=100))['status']=='INVALID'
    assert set(range(55,101,5)) <= measurement_tasks(100)
    assert 2*len([t for t in measurement_tasks(200) if t>100]) >= 20
    write_json(tmp_path/'summary.json', {'device': torch.device('cpu')})
    assert json.loads((tmp_path/'summary.json').read_text()) == {'device':'cpu'}


def test_auto_scaling():
    from methods.vna_output import choose_solver
    assert choose_solver('auto',4096)[0]=='output_exact'
    assert choose_solver('auto',25600)[0]=='cg'
    assert choose_solver('auto',13,12)[0]=='cg'
    assert choose_solver('output_exact',25600)[0]=='output_exact'


def test_mnist_first_jvp_through_production_zeros_patch(tmp_path):
    # Fresh interpreter: no fixture may pre-load JVP decompositions.
    code = '''
import sys
assert 'torch._decomp.decompositions_for_jvp' not in sys.modules
from experiments.stage_b5 import upstream_imports, run_native
import torch
from pathlib import Path
from adapters.lop_hooks import LearnHook
from tests.test_vna_stage_b5 import fixture, actual_measure
upstream_imports()
torch.set_num_threads(1)
original = torch.zeros
params = fixture('mnist','bp',Path(sys.argv[1]),tasks=1)
def measured(model,batch,spec,ctx,*,generator):
    assert torch.zeros is not original
    assert torch.zeros((0,2)).shape == (1,2)
    return actual_measure(model,batch,spec,ctx,generator=generator)
hook = LearnHook('mnist','bp',params,measured)
run_native('mnist',params,Path(sys.argv[1]),hook)
assert len(hook.rows)==2
assert torch.zeros is original
'''
    result = subprocess.run([sys.executable,'-c',code,str(tmp_path)],
        env={**os.environ,'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION':'python','TQDM_DISABLE':'1'},capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('spec', [CE(), KDRepDistiller(), MSE()])
def test_whole_point_exact_cg(spec):
    from tests.test_measure_efficiency import SmallResNet, sites
    from diagnostics.icarl_measure import B3Measurement
    model = SmallResNet().double().eval()
    batch = torch.randn(8,3,32,32,dtype=torch.float64), torch.arange(8)%3
    if isinstance(spec, MSE): batch = batch[0], torch.randn(8,3,dtype=torch.float64)
    def context(c,b):
        return spec, dict(targets=b[1],teacher_logits=torch.zeros(len(b[0]),3,dtype=torch.float64))
    ctx=dict(task=2,epoch=1,step=1,end_of_task=False,train_acc=.5,known_classes=1)
    records=[]
    for solver in ('output_exact','cg'):
        measure=B3Measurement(dict(seed=3,setting='B3'),sites_fn=sites,context_fn=context,
            ranks=(1,),groups=False,solver=solver,cg_tol=1e-12,cg_iters=2000)
        records.append(measure(model,batch,spec,ctx,generator=torch.Generator().manual_seed(7)))
    for exact,cg in zip(*records):
        for key in ('V','rho','rho_joint','nu'):
            assert exact[key] == pytest.approx(cg[key],rel=1e-8,abs=1e-25)
        for kind in ('N1','N2'):
            for key in ('rho','V','nu','e_norm2'):
                torch.testing.assert_close(torch.tensor(json.loads(exact[f'null_{key}_{kind}']), dtype=torch.float64),
                    torch.tensor(json.loads(cg[f'null_{key}_{kind}']), dtype=torch.float64),rtol=1e-8,atol=1e-25)


def test_b3_schedule_and_nonempty_cbp_lowest_utility(monkeypatch):
    from adapters.icarl_hooks import measurement_epoch
    from adapters.lop_hooks import Prediction
    from tests.test_measure_efficiency import Native
    from diagnostics import lop_measure as mod
    points = [(task, epoch) for task in range(1,11)
              for epoch in range(1,(200 if task==1 else 170)+1)
              if measurement_epoch(task,epoch,200 if task==1 else 170)]
    assert len(points)==13
    assert [(task,epoch) for task,epoch in points if epoch in (170,200)] == [(2,170),(5,170),(10,170)]
    model=Prediction(Native()).double().eval()
    batch=torch.randn(8,3,dtype=torch.float64),torch.arange(8)%3
    original=mod.activity_stats
    def active(m,b):
        stats=original(m,b)
        for r in stats: r['dormant_indices']=torch.empty(0,dtype=torch.long)
        return stats
    monkeypatch.setattr(mod,'activity_stats',active)
    shadow=SimpleNamespace(bounds=[.1],bias_corrected_util=[torch.tensor([10.,1.])])
    ctx=dict(task=1,step=8,point='first',within_task_step=8,buffer_task_purity=1.,
        shadow=shadow,cbp_candidates=[('cbp',shadow,[torch.empty(0,dtype=torch.long)])],
        init_row_norms=[1.],loss_fn=lambda z,c: nn.functional.cross_entropy(z,c['targets']))
    rows=mod.B5Measurement('mnist',dict(seed=4),ranks=[1])(model,batch,CE(),ctx,
        generator=torch.Generator().manual_seed(7))
    assert all(r['cbp_m']==1 and r['redo_m']==0 and r['cbp_selection']=='lowest_utility_m' for r in rows)
