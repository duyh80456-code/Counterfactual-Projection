"""Profile comparison must expose numerical differences and missing results."""
from copy import deepcopy
import inspect
from diagnostics.exact_solver import compare_profiles,curvature_differences
from experiments.main_cil import measure_task


def test_curvature_configuration_defaults_and_checkpoint_state():
    import pytest
    from methods.vna_v3 import V3Engine
    from experiments.main_cil import parser
    assert parser().parse_args(['--out','unused']).curvature=='exact'
    assert parser().parse_args(['--out','unused','--curvature','power10']).curvature=='power10'
    restored=V3Engine(curvature='power10');restored.load_state_dict(V3Engine(curvature='power10').state_dict())
    assert restored.curvature=='power10'
    with pytest.raises(ValueError,match='curvature'):
        restored.configure_schedule(dict(K0=1,curvature='exact'))
    with pytest.raises(ValueError,match='curvature'): V3Engine(curvature='invalid')


def test_profile_comparison_reports_curvature_selection_changes():
    report=dict(status='PROFILE_COMPLETED',application_seconds=1230.,final_L_KD=[2.,1.],curvature='power10',
        identity=dict(checkpoint_sha256='fixture'),
        applied_swaps=[dict(site='s',unit=0,candidate_id='a',A=dict(weight=[1.]))],
        candidate_rankings=[dict(site='s',candidates=[dict(id='a',score=1.,curvature=1.,power_generator_seed=2**60,power_iterations=10),
            dict(id='b',score=.5,curvature=2.,power_generator_seed=12,power_iterations=10)])],
        applications=[dict(mu_star=1e-7,rho_tr=.5,L_before=3.,L_after=2.,KD_before=2.,KD_after=1.,
            swap_records=[dict(site='s',unit=0)],
            gir_trials=[dict(site='s',unit=0,Gamma_woodbury=.1)])])
    same=compare_profiles(report,deepcopy(report))
    assert same['status']=='COMPARED' and all(row['matches'] for row in same['checks'].values())
    assert compare_profiles(report,deepcopy(report),strict=True)['status']=='PASS'
    wrong_seed=deepcopy(report);wrong_seed['candidate_rankings'][0]['candidates'][0]['power_generator_seed']+=1
    assert compare_profiles(report,wrong_seed,strict=True)['status']=='FAIL'
    nonfinite=deepcopy(report);nonfinite['applications'][0]['gir_trials'][0]['Gamma_woodbury']=float('inf')
    assert compare_profiles(nonfinite,deepcopy(nonfinite),strict=True)['status']=='FAIL'
    current=deepcopy(report);current['application_seconds']=120.
    current['candidate_rankings'][0]['candidates'].reverse()
    current['applications'][0]['gir_trials'][0]['Gamma_woodbury']=.2
    current['final_L_KD'][0]=1.5
    result=compare_profiles(report,current)
    assert result['checks']['mu_star']['matches']
    assert not result['checks']['candidate_rankings']['matches']
    assert not result['checks']['ordered_Gamma']['matches']
    assert not result['checks']['final_L_KD']['matches']
    assert compare_profiles(report,current,strict=True)['status']=='FAIL'
    current['curvature']='exact'
    changes=compare_profiles(report,current)['ranking_changes']
    assert changes['calls'][0]['valid_count']==2
    assert changes['calls'][0]['candidates'][0]['delta']==0.
    assert 'independent trajectories' in changes['context']
    assert compare_profiles(report,dict(status='CUDA_OOM'))['status']=='INCOMPLETE'


def test_curvature_audit_reports_errors_top_k_and_kendall_tau_b():
    candidates=[dict(id=str(i),curvature=float(i),curvature_power10=float(4-i),
        score=1/i,score_power10=1/(4-i)) for i in (1,2,3)]
    result=curvature_differences(dict(candidate_rankings=[dict(site='s',candidates=candidates)]))['calls'][0]
    assert result['kendall_tau_b']==-1.
    assert result['top_k_overlap']['1']==dict(k=1,common=0,fraction=0.)
    assert result['top_k_overlap']['2']==dict(k=2,common=1,fraction=.5)
    assert result['candidates'][0]['delta']==-2.
    assert result['candidates'][0]['relative_error']==2.


def test_main_task_log_has_no_retired_reach():
    assert 'reach' not in inspect.getsource(measure_task)


def test_capacity_proxy_isolates_model_rng_and_records_oom(monkeypatch):
    import torch
    from torch import nn
    from diagnostics.exact_solver import profile_capacity
    from diagnostics.vna_align import _assert_bitwise
    from experiments.vna_resume import rng_state
    from methods import vna_v3
    class Native(nn.Module):
        def __init__(self):
            super().__init__(); self.fc=nn.Linear(2,10).double()
        def update_fc(self,n): self.fc=nn.Linear(2,n)
    model=nn.Module();model.native=Native()
    before=deepcopy(model.state_dict()),rng_state()
    seen=[]
    def diagnose(clone,batch,**kw):
        assert clone is not model and clone.native.fc.out_features==100
        assert clone.native.fc.weight.dtype==torch.float64
        seen.append(len(batch[0]))
        if len(batch[0])==256: raise torch.cuda.OutOfMemoryError('capacity fixture')
        return dict(solver='cg',gram_bytes=0,m=12800,solves=[dict(converged=True)])
    monkeypatch.setattr(vna_v3,'diagnose',diagnose)
    monkeypatch.setattr('adapters.icarl_hooks.sites_for',lambda _:[])
    for name in ('synchronize','reset_peak_memory_stats','empty_cache'):
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ('max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda:2048)
    result=profile_capacity(model,(torch.ones(128,2,dtype=torch.float64),torch.zeros(128,dtype=torch.long)))
    assert seen==[256,128] and [r['status'] for r in result['rows']]==['CUDA_OOM','COMPLETED']
    assert result['rows'][1]['solver']=='cg' and result['rows'][1]['gram_bytes']==0
    assert not result['actual_task20_measured'] and not result['probe_change_applied']
    _assert_bitwise(before,(model.state_dict(),rng_state()))


def test_lm_audit_distinguishes_numeric_failure_from_measured_model_rejection():
    report=dict(status='PROFILE_COMPLETED',curvature='power10',application_seconds=1.,final_L_KD=[1.,0.],
        applications=[dict(mu_star=1.,rho_tr=.5,L_before=2.,L_after=1.,KD_before=0.,KD_after=0.,
            mu_trajectory=[.01,.1,1.],rho_trajectory=[-1.,.2,.5])])
    current=deepcopy(report)
    current['applications'][0]['lm_trials']=[dict(mu=.01,outcome='numeric_failure',failure_reason='ir_stagnation'),
        dict(mu=.1,outcome='numeric_failure',failure_reason='inner_budget'),
        dict(mu=.3,outcome='numeric_failure',failure_reason='inner_budget')]
    audit=compare_profiles(report,current)['lm_numeric_audit']
    assert [r['mu'] for r in audit['positive_reference_rejections']]==[.1]
    assert audit['unknown_reference_trials']==[.3]


def test_capacity_proxy_includes_double_stage1_downstream_with_native_graphs(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import torch
    from torch import nn
    from methods import vna_v3 as vna
    from diagnostics.exact_solver import profile_capacity
    class Native(nn.Module):
        def __init__(self):super().__init__();self.fc=nn.Linear(2,10).double()
        def update_fc(self,n):self.fc=nn.Linear(2,n)
    model=nn.Module();model.native=Native();seen=[]
    @contextmanager
    def products(*args,**kw):yield 'products'
    graph=SimpleNamespace(JT=lambda seed:seen.append(('low',len(seed))),
        high=SimpleNamespace(JT=lambda seed:seen.append(('high',len(seed))),products=SimpleNamespace(model='double')))
    def downstream(model,site,x):
        assert model=='double' and site=='stage1' and len(x)==128 and x.dtype==torch.float64
        return SimpleNamespace(bytes=1234,seconds=1.,vjp_calls=101,dtype=torch.float64,
            block_samples=8,blocks=16,measure=lambda:seen.append(('stream',8)))
    monkeypatch.setattr(vna,'training_products',products)
    monkeypatch.setattr(vna,'FrozenGram',lambda _:graph)
    monkeypatch.setattr(vna,'Downstream',downstream)
    monkeypatch.setattr(vna,'diagnose',lambda *a,**kw:dict(diagnostic_flag=None))
    monkeypatch.setattr('adapters.icarl_hooks.sites_for',lambda _:['stage1'])
    for name in ('synchronize','reset_peak_memory_stats','empty_cache'):monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ('max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda:2048)
    result=profile_capacity(model,(torch.ones(128,2,dtype=torch.float64),torch.zeros(128,dtype=torch.long)))
    assert seen==[('low',12800),('high',12800),('stream',8)]
    assert result['rows'][1]['downstream_capacity']['status']=='MEASURED'
    assert result['rows'][1]['downstream_capacity']['bytes']==1234
