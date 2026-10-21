"""Profile comparison must expose numerical differences and missing results."""
from copy import deepcopy
import inspect
from diagnostics.exact_solver import compare_profiles
from experiments.main_cil import measure_task


def test_profile_comparison_reports_curvature_selection_changes():
    report=dict(status='PROFILE_COMPLETED',application_seconds=1230.,final_L_KD=[2.,1.],
        candidate_rankings=[dict(candidates=[dict(id='a'),dict(id='b')])],
        applications=[dict(mu_star=1e-7,rho_tr=.5,L_before=3.,L_after=2.,KD_before=2.,KD_after=1.,
            swap_records=[dict(site='s',unit=0)],
            gir_trials=[dict(site='s',unit=0,Gamma_woodbury=.1)])])
    same=compare_profiles(report,deepcopy(report))
    assert same['status']=='COMPARED' and all(row['matches'] for row in same['checks'].values())
    current=deepcopy(report);current['application_seconds']=120.
    current['candidate_rankings'][0]['candidates'].reverse()
    current['applications'][0]['gir_trials'][0]['Gamma_woodbury']=.2
    current['final_L_KD'][0]=1.5
    result=compare_profiles(report,current)
    assert result['checks']['mu_star']['matches']
    assert not result['checks']['candidate_rankings']['matches']
    assert not result['checks']['ordered_Gamma']['matches']
    assert not result['checks']['final_L_KD']['matches']
    assert compare_profiles(report,dict(status='CUDA_OOM'))['status']=='INCOMPLETE'


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
    def fake_gram(_): return torch.ones(1),dict(gram_seconds=59.,m=seen[-1]*100)
    def diagnose(clone,batch,**kw):
        assert clone is not model and clone.native.fc.out_features==100
        assert clone.native.fc.weight.dtype==torch.float64
        seen.append(len(batch[0]))
        vna_v3.gram(None)
        if len(batch[0])==256: raise torch.cuda.OutOfMemoryError('capacity fixture')
        return dict(gram_seconds=59.,m=12800)
    monkeypatch.setattr(vna_v3,'diagnose',diagnose)
    monkeypatch.setattr(vna_v3,'gram',fake_gram)
    monkeypatch.setattr('adapters.icarl_hooks.sites_for',lambda _:[])
    for name in ('synchronize','reset_peak_memory_stats','empty_cache'):
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ('max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda:2048)
    result=profile_capacity(model,(torch.ones(128,2,dtype=torch.float64),torch.zeros(128,dtype=torch.long)))
    assert seen==[256,128] and [r['status'] for r in result['rows']]==['CUDA_OOM','COMPLETED']
    assert vna_v3.gram is fake_gram
    assert result['rows'][0]['gram_status']=='COMPLETED' and result['rows'][0]['gram_seconds']==59.
    assert not result['actual_task20_measured'] and not result['probe_change_applied']
    _assert_bitwise(before,(model.state_dict(),rng_state()))
