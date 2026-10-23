"""The shared clip leaves inactive gradients untouched and sees effective updates."""
import copy
from types import SimpleNamespace
import pytest
import torch
from experiments.grad_clip import GradClip,upgd_step,optimizer_clipping
from adapters.v4_competitors import source_module,cflat_module


@pytest.mark.parametrize('tau',[None,1e20])
def test_sgd_inactive_clip_is_bitwise_identity(tau):
    torch.manual_seed(17)
    initial=torch.nn.Linear(3,2).double();x=torch.randn(5,3,dtype=torch.float64);y=torch.randn(5,2,dtype=torch.float64)
    results=[]
    for monitor in (False,True):
        net=copy.deepcopy(initial);opt=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9,weight_decay=.001)
        clip=GradClip(tau);losses=[]
        for step in range(5):
            opt.zero_grad();loss=(net(x)-y).square().mean();loss.backward()
            before=[(p.grad,p.grad.clone(),p.grad._version) for p in net.parameters()]
            if monitor:
                clip.apply((p.grad for p in net.parameters()),task=1,step=step+1)
                for p,(original,value,version) in zip(net.parameters(),before):
                    assert p.grad is original and p.grad._version==version and torch.equal(p.grad,value)
            opt.step();losses.append(loss.detach().clone())
        results.append((net.state_dict(),losses))
    for k in results[0][0]:assert torch.equal(results[0][0][k],results[1][0][k])
    assert all(torch.equal(a,b) for a,b in zip(results[0][1],results[1][1]))


def test_small_clip_bounds_norm_and_stats_resume():
    grads=[torch.tensor([3.,4.],dtype=torch.float64),torch.tensor([12.],dtype=torch.float64)]
    clip=GradClip(2.);clip.apply(grads,task=2,step=10)
    assert torch.cat(grads).norm()<=2.
    report=clip.report();assert report['run']['n_clipped']==1 and report['run']['max_preclip_norm']==13.
    resumed=GradClip(2.);resumed.load_state_dict(clip.state_dict());assert resumed.report()==report
    with pytest.raises(ValueError):GradClip(1.).load_state_dict(clip.state_dict())


@pytest.mark.parametrize('tau',[None,1e20])
def test_upgd_inactive_clip_matches_pinned_native_weights_state_and_rng(tau):
    native=source_module('clip_native_upgd','upgd/core/optim/weight_upgd/first_order.py').FirstOrderGlobalUPGD
    results=[]
    for monitored in (False,True):
        torch.manual_seed(42);p=torch.nn.Parameter(torch.tensor([-.5,.7],dtype=torch.float64))
        opt=native([('weight',p)],lr=.05,beta_utility=.999,sigma=.001,weight_decay=.001)
        clip=GradClip(tau)
        for step in range(4):
            p.grad=torch.tensor([.3,-.7],dtype=p.dtype)
            if monitored:upgd_step(opt,lambda o,f,g:clip.apply(g,task=2,step=step+1))
            else:opt.step()
        results.append((p.detach().clone(),copy.deepcopy(opt.state[p]),torch.get_rng_state()))
    assert torch.equal(results[0][0],results[1][0]) and torch.equal(results[0][2],results[1][2])
    assert results[0][1]['step']==results[1][1]['step']
    assert torch.equal(results[0][1]['avg_utility'],results[1][1]['avg_utility'])


def test_upgd_clips_after_noise_and_utility():
    native=source_module('clip_native_upgd','upgd/core/optim/weight_upgd/first_order.py').FirstOrderGlobalUPGD
    p=torch.nn.Parameter(torch.tensor([-.5,.7],dtype=torch.float64));p.grad=torch.tensor([3.,-7.],dtype=p.dtype)
    opt=native([('weight',p)],lr=.05,beta_utility=.999,sigma=.001);raw=p.grad.clone();seen=[]
    def callback(o,f,g):
        seen.append(torch.cat(g).clone());GradClip(.1).apply(g,task=2,step=1)
        assert torch.cat(g).norm()<=.1
    upgd_step(opt,callback)
    assert not torch.equal(seen[0],raw) and torch.equal(p.grad,raw)


def test_cflat_clips_only_aggregated_base_gradient():
    from experiments.runners.pycil import upstream_imports
    upstream_imports();cflat_module()
    from optims.c_flat import C_Flat
    net=torch.nn.Linear(2,1).double();proxy=SimpleNamespace(optim=torch.optim);seen=[]
    clip=GradClip(.01)
    def callback(opt,frame):
        seen.append([p.grad.detach().clone() for p in net.parameters()])
        clip.apply((p.grad for p in net.parameters()),task=2,step=len(seen))
    with optimizer_clipping(proxy,callback):
        base=proxy.optim.SGD(net.parameters(),lr=.1)
    opt=C_Flat(net.parameters(),base,net,cflat=True)
    x=torch.tensor([[1.,2.],[2.,1.]],dtype=torch.float64)
    def loss():return net(x),[net(x).square().mean()]
    opt.set_closure(loss);opt.step()
    assert len(seen)==1 and clip.report()['run']['n_clipped']==1
    assert torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in net.parameters()]))<=.01


def test_vna_intervention_remains_outside_sgd_clip():
    # The main hook invokes the callback only at SGD; a geometric edit isn't a gradient.
    p=torch.nn.Parameter(torch.tensor([1.]));p.grad=torch.tensor([10.]);clip=GradClip(.1)
    optimizer=torch.optim.SGD([p],lr=1.)
    optimizer.register_step_pre_hook(lambda o,a,k:clip.apply([p.grad],task=2,step=1) and None)
    optimizer.step();before=clip.report()['run']['n_steps']
    with torch.no_grad():p.add_(torch.tensor([50.]))
    assert clip.report()['run']['n_steps']==before and float(p)>50.


def test_nonfinite_norm_is_not_reported_as_finite_max():
    clip=GradClip(None)
    clip.apply([torch.tensor([2.])],task=1,step=1)
    clip.apply([torch.tensor([float('inf')])],task=1,step=2)
    stats=clip.report()['run']
    assert stats['max_preclip_norm'] is None and stats['nonfinite_steps']==1
    assert stats['max_finite_preclip_norm']==2. and stats['preclip_norm']['finite_quantiles_only']


def test_study_restore_excludes_sources_for_loose_and_archive(tmp_path):
    import tarfile
    from experiments.vna_kaggle import restore_stage_archives
    source=tmp_path/'input';stage=source/'clip_study'/'commit'
    (stage/'_sources').mkdir(parents=True)
    (stage/'_sources'/'latest.pt').write_bytes(b'old checkpoint')
    (stage/'study.json').write_text('{}')
    archive=source/'study.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:stream.add(stage.parent,arcname='outputs/clip_study')
    target=tmp_path/'out'
    restore_stage_archives(source,target,'clip_study',exclude_parts=('_sources',))
    assert (target/'commit'/'study.json').read_text()=='{}'
    assert not (target/'commit'/'_sources').exists()


def test_clip_study_notebooks_have_isolated_inputs_and_selection_gate():
    from scripts.build_kaggle_vna_parallel import build_clip_study
    for phase in ('AB','C'):
        notebook=build_clip_study('a'*40,phase)
        sources=[''.join(cell['source']) for cell in notebook['cells']]
        for cell,text in zip(notebook['cells'],sources):
            if cell['cell_type']=='code':compile(text,f'clip_{phase}','exec')
        source='\n'.join(sources)
        assert "exclude_parts=('_sources',)" in source
        assert '--phase' in source and f"'{phase}'" in source
        assert 'restore_stage_archives' in source and 'VNA_DEADLINE' in source
        if phase=='C':assert 'load_selection' in source


def test_archive_retains_previous_when_temporary_space_is_insufficient(tmp_path,monkeypatch):
    from experiments import vna_kaggle
    root=tmp_path/'run';(root/'outputs').mkdir(parents=True)
    (root/'outputs'/'report.json').write_bytes(b'12345678')
    old=root.with_suffix('.tar.gz');old.write_bytes(b'previous archive')
    monkeypatch.setattr(vna_kaggle.shutil,'disk_usage',lambda p:SimpleNamespace(free=1))
    with pytest.raises(RuntimeError,match='ARCHIVE STALE'):
        vna_kaggle.archive_outputs(root,folders=('outputs',))
    assert old.read_bytes()==b'previous archive'
