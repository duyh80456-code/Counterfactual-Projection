"""H3 bounded search and H2 real-loader gates; fixtures are never benchmark results."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from diagnostics.vna_align import _global_state,_assert_bitwise


@pytest.fixture(autouse=True)
def isolate():
    threads=torch.get_num_threads();torch.set_num_threads(1)
    with _global_state():yield
    torch.set_num_threads(threads)


def test_h3_small_overrides_and_required_prefix_keep_upstream_unchanged():
    from experiments.standard_imagenet import effective_params,required_classes
    from experiments.runners import lop
    cfg=json.loads(Path('configs/standard_cl/h3_imagenet_small.json').read_text())
    source=lop.UPSTREAM/cfg['upstream_config'];before=source.read_bytes()
    params,overrides=effective_params(cfg)
    assert (params['num_tasks'],params['num_showings'],params['mini_batch_size'])==(100,50,100)
    assert overrides==dict(num_tasks=100,num_showings=50)
    assert cfg['intervention_every']==4000 and cfg['window']==128 and cfg['max_candidates']==16
    used=required_classes(0,params['num_tasks']);assert len(used)==len(set(used))==200
    assert np.array_equal(used,required_classes(0,5000)[:200])
    assert effective_params(cfg,num_tasks=2,num_showings=1)[1]==dict(num_tasks=2,num_showings=1)
    full=json.loads(Path('configs/standard_cl/h3_imagenet.json').read_text())
    assert effective_params(full)[0]['num_tasks']==5000 and effective_params(full)[0]['num_showings']==250
    assert source.read_bytes()==before


def test_h3_hashes_only_required_original_files_and_validates_them(tmp_path):
    from experiments.standard_imagenet import data_identity
    for c in (7,19):np.save(tmp_path/f'{c}.npy',np.zeros((700,3,32,32),dtype=np.uint8))
    (tmp_path/'999.npy').write_bytes(b'ignored unrelated malformed file')
    identity=data_identity(tmp_path,[19,7,19]);assert set(identity)=={'7','19'}
    with pytest.raises(FileNotFoundError):data_identity(tmp_path,[8])
    np.save(tmp_path/'7.npy',np.zeros((699,3,32,32),dtype=np.uint8))
    with pytest.raises(ValueError,match='malformed'):data_identity(tmp_path,[7])


def test_preliminary_lm_score_is_actual_parameter_displacement():
    from methods.cl_h3 import preliminary_score
    from methods.cl_state import vector,State
    from methods.vna_sites import HookSite
    from methods.vna_v3 import V3Engine
    model=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,1)).double()
    site=HookSite(model,'0','2','linear',between=('1',));optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    torch.manual_seed(4);gradients={id(p):torch.randn_like(p) for p in model.parameters()}
    flat=torch.cat([gradients[id(p)].flatten() for p in model.parameters()]);initial=vector(model);snapshot=State(model,optimizer)
    proposal={'A':dict(weight=torch.ones(1,2,dtype=torch.double),bias=torch.zeros(1,dtype=torch.double))}
    score=preliminary_score(site,proposal,1,gradients,.2)
    V3Engine('vna').install_gir(model,site,[proposal],[1],optimizer)
    delta=vector(model)-initial
    assert score==pytest.approx(float(-flat@delta-.1*delta.square().sum()),abs=1e-12)
    snapshot.restore()


def test_bounded_h3_exact_search_limit_identity_determinism_and_rollback(monkeypatch):
    from methods.cl_h3 import TangentReconfiguration
    from methods.cl_replay import CorrectionConfig,LossComponent
    from methods.vna_losses import MSE
    from methods.vna_sites import HookSite
    from methods.cl_state import State
    from experiments.vna_resume import rng_state
    torch.manual_seed(8);net=nn.Sequential(nn.Linear(2,20),nn.ReLU(),nn.Linear(20,1)).double()
    with torch.no_grad():net[0].bias.fill_(1.)
    optimizer=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9)
    x=torch.randn(4,2,dtype=torch.double);y=torch.zeros(4,1,dtype=torch.double)
    losses=SimpleNamespace(targets=y,spec=MSE(),ctx=dict(targets=y),components=(LossComponent('mse',lambda z:(z-y).square().mean()),LossComponent('zero',lambda z:z.sum()*0)))
    site=HookSite(net,'0','2','linear',between=('1',));snapshot=State(net,optimizer)
    cfg=CorrectionConfig(damping_relative=1.,cg_iterations=150,cg_tolerance=1e-7,trace_probes=1)
    engine=TangentReconfiguration(cfg,max_candidates=16);event=engine.step(net,optimizer,x,losses,[site])
    assert event['generated_candidates']>16 and event['exact_candidate_evaluations']<=16
    assert event['identity_evaluations']==1 and len(event['ranking'])==event['generated_candidates']
    assert sum(r['selected'] for r in event['ranking'])==16
    assert 'shortlist' in event['search_scope'] and event['choice'] in {'identity',*(r['candidate'] for r in event['trials'])}
    first=deepcopy(net.state_dict());first_rng=rng_state();snapshot.restore()
    repeat=TangentReconfiguration(cfg,max_candidates=16).step(net,optimizer,x,losses,[site])
    assert repeat['ranking']==event['ranking'] and repeat['choice']==event['choice']
    _assert_bitwise(first,net.state_dict(),'deterministic winner');_assert_bitwise(first_rng,rng_state(),'winner rng')
    with pytest.raises(ValueError,match='candidate search'):TangentReconfiguration(cfg,max_candidates=15).load_state_dict(engine.state_dict())
    snapshot.restore()
    from methods.cl_replay import ReplayCorrection
    def pause():raise SystemExit(75)
    monkeypatch.setattr(ReplayCorrection,'boundary',staticmethod(pause))
    with pytest.raises(SystemExit):TangentReconfiguration(cfg,max_candidates=16).step(net,optimizer,x,losses,[site])
    _assert_bitwise(snapshot.values,net.state_dict(),'rollback')


def test_real_h2_replay_origin_and_geometry_gates():
    pytest.importorskip('avalanche')
    from diagnostics.standard_preflight import replay_gate,geometry_gate
    assert replay_gate()['gradient_decomposition']
    assert geometry_gate()['actual_projection_changed']


def test_h2_is_pinned_official_readme_configuration():
    from methods.cl_lpr import ROOT
    cfg=json.loads(Path('configs/standard_cl/h2_online_cifar100.json').read_text())['lpr']
    assert (cfg['preconditioner']['omega_0'],cfg['preconditioner']['beta'],cfg['update']['every_iter'])==(4.,2.,30)
    source=(ROOT/'README.md').read_text()
    assert 'strategy.lpr_kwargs.preconditioner.omega_0=4 strategy.lpr_kwargs.preconditioner.beta=2 strategy.lpr_kwargs.update.every_iter=30' in source


def test_automatic_official_dataset_download_when_absent(tmp_path,monkeypatch):
    import torchvision.datasets as datasets
    from diagnostics.standard_preflight import prepare_data
    calls=[]
    class Fixture:
        def __init__(self,root,train,download):
            calls.append((train,download));self.targets=np.tile(np.arange(100),500 if train else 100)
        def __len__(self):return len(self.targets)
    monkeypatch.setattr(datasets,'CIFAR100',Fixture)
    assert prepare_data('H1',tmp_path/'absent',tmp_path/'stage',{})==str(tmp_path/'stage')
    assert calls==[(True,True),(False,True)]


def test_notebook_pairs_have_same_native_budget_and_preflight():
    import ast
    from scripts.build_kaggle_standard_compare import build,jobs
    for branch in ('H1','H2','H3','H4'):
        lanes=jobs(branch);left,right=lanes[0][0],lanes[1][0]
        assert left['module']==right['module']
        def native(args):
            index=args.index('--arm');return args[:index]+args[index+2:]
        assert native(left['args'])==native(right['args'])
        nb=build(branch,'f'*40);text=''.join(''.join(c['source']) for c in nb['cells'])
        assert 'diagnostics.standard_preflight' in text and 'PREFLIGHT PASSED' in text and 'export_comparison' in text
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
    h3=str(jobs('H3'));assert 'h3_imagenet_small.json' in h3 and 'h3_imagenet.json' not in h3


def test_export_keeps_paused_run_partial_and_writes_curves(tmp_path):
    import csv
    from experiments.io_utils import atomic_json_save
    from scripts.collect_standard_cl import collect,export_comparison
    for arm,status in (('cbp','COMPLETED'),('vna_gir','PAUSED')):
        out=tmp_path/arm;out.mkdir()
        protocol=dict(branch='H3',arm=arm,seed=0,params={},config={},tasks=2,data_sha256={'fixture':'test only'})
        tasks=[dict(task=i+1,final_accuracy=90.,mean_error=10.,learning_speed=5.) for i in range(2 if status=='COMPLETED' else 1)]
        summary=dict(protocol=protocol,status=status,tasks=tasks,last_50_task_error=10.,plasticity_decay=0.,final_accuracy=90.,
            gpu_hours=0.,wall_seconds=1.,correction_seconds=0.,peak_vram_bytes=0,actual_swaps=0,candidate_search={})
        atomic_json_save(protocol,out/'protocol.json');atomic_json_save(summary,out/'summary.json')
    grouped=collect([tmp_path]);assert next(r for r in grouped if r['arm']=='vna_gir')['metrics']['last_50_task_error']['mean'] is None
    rows=export_comparison(tmp_path);assert next(r for r in rows if r['status']=='PAUSED')['final_accuracy'] is None
    assert (tmp_path/'accuracy_curves.png').is_file()
    with (tmp_path/'curves.csv').open() as stream:assert len(list(csv.DictReader(stream)))==3
    path=tmp_path/'vna_gir/summary.json';bad=json.loads(path.read_text());bad['status']='COMPLETED';atomic_json_save(bad,path)
    with pytest.raises(ValueError,match='incomplete completed'):export_comparison(tmp_path)


def test_uncertified_candidate_keeps_partial_audit_and_rolls_back():
    from methods.cl_h3 import choose_swap,SwapCandidate
    from methods.cl_projection import Uncertified
    from methods.cl_state import State
    model=nn.Linear(1,1,bias=False).double();model.weight.data.fill_(1.)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1);snapshot=State(model,optimizer)
    def install(net):net.weight.data.fill_(.5)
    candidates=[SwapCandidate(name,'live',install,lambda *a:None) for name in ('first','failed')]
    calls=0
    def objective(net):
        nonlocal calls
        calls+=1
        if calls==3:raise Uncertified('fixture CG failure',[])
        return float(net.weight.square().sum())
    with pytest.raises(Uncertified) as exc:
        choose_swap(model,optimizer,torch.ones(1,1,dtype=torch.double),(lambda z:z.square().mean(),),candidates,surrogate=objective)
    assert exc.value.failed_candidate=='failed' and len(exc.value.candidate_trials)==1
    _assert_bitwise(snapshot.values,model.state_dict(),'uncertified rollback')
