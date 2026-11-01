"""Only T1–T10 from the ETT specification, float64 CPU."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from torch import nn
from ett.core.units import LinearUnitLayer,ConvUnitLayer
from ett.core.expand import candidates,expand
from ett.core.rebuild import rebuild
from ett.core.stats import design,gram,measurement
from ett.core.compensate import compensate,g_exact,columns
from ett.core.select import backward
from ett.core.ett import ETT
from ett.optim.adam_rowstep import AdamRowStep


@pytest.fixture(autouse=True)
def deterministic():
    old=torch.get_num_threads();state=torch.get_rng_state();torch.set_num_threads(1);torch.manual_seed(3)
    yield
    torch.set_rng_state(state);torch.set_num_threads(old)


def pair(conv=False,bn=True):
    a=nn.Conv2d(2,3,3,padding=1) if conv else nn.Linear(2,3)
    b=nn.Conv2d(3,2,3,padding=1) if conv else nn.Linear(3,2)
    norm=nn.BatchNorm2d(3) if conv else nn.BatchNorm1d(3)
    net=nn.Sequential(a,norm if bn else nn.Identity(),nn.ReLU(),b).double()
    cls=ConvUnitLayer if conv else LinearUnitLayer
    site=cls(net,'0','3','2','1' if bn else None)
    x=torch.randn(8,2,5,5,dtype=torch.double) if conv else torch.randn(8,2,dtype=torch.double)
    y=torch.randn(8,2,5,5,dtype=torch.double) if conv else torch.randn(8,2,dtype=torch.double)
    return net,site,x,y


def output(net,x,training):
    with measurement(net,training),torch.no_grad():return net(x).clone()


@pytest.mark.parametrize('conv',[False,True])
@pytest.mark.parametrize('bn',[False,True])
def test_T1_expand_preserves_function(conv,bn):
    net,site,x,y=pair(conv,bn);opt=torch.optim.SGD(net.parameters(),lr=.01)
    before=[output(net,x,t) for t in (False,True)]
    proposal=candidates(site,x,y,nn.functional.mse_loss,2,torch.Generator().manual_seed(9))
    expand(site,opt,proposal)
    for training,ref in zip((False,True),before):assert float((output(net,x,training)-ref).abs().max())<1e-12


@pytest.mark.parametrize('conv',[False,True])
def test_T2_dead_units_exact_train_eval(conv):
    net,site,x,y=pair(conv,True);opt=torch.optim.SGD(net.parameters(),lr=.01)
    with torch.no_grad():site.bn.weight[0]=0;site.bn.bias[0]=-1
    before=[output(net,x,t) for t in (False,True)]
    W=site.weights();keep=[1,2];idx=columns(keep,site.group_size)+[len(W)-1]
    rebuild(site,opt,keep,W[idx])
    for t,ref in zip((False,True),before):assert float((output(net,x,t)-ref).abs().max())<1e-12
    # C1 also handles a wholly zero feature Gram without inventing a ridge floor.
    with torch.no_grad():site.bn.weight.zero_();site.bn.bias.fill_(-1)
    ref=output(net,x,False);engine=ETT(net,opt,[site],spatial_rows={site.incoming:25} if conv else None)
    engine.active=dict(original={site.incoming:1},trained=0,expand_seconds=0.,train_seconds=0.)
    while len(engine.buffer)<engine.buffer.capacity:engine.buffer.append((x,y))
    engine.ages[site.incoming].fill_(engine.maturity)
    records=engine.contract(nn.functional.mse_loss)
    assert records[0]['exact_dead_path'] and records[0]['cholesky_attempts']==[]
    assert float((output(net,x,False)-ref).abs().max())<1e-12


@pytest.mark.parametrize('conv',[False,True])
def test_T3_preactivation_error_is_removed_contribution(conv):
    net,site,x,y=pair(conv,False);opt=torch.optim.SGD(net.parameters(),lr=.01)
    with torch.no_grad():feature=net[2](net[1](net[0](x)));A=design(site,feature);W=site.weights();Z=A@W
    keep=[1,2];idx=columns(keep,site.group_size)+[len(W)-1]
    removed=A[:,columns([0],site.group_size)]@W[columns([0],site.group_size)]
    rebuild(site,opt,keep,W[idx])
    with torch.no_grad():new=net(x)
    flat=new.permute(0,2,3,1).reshape(-1,2) if conv else new
    assert float((flat-Z+removed).abs().max())<1e-12


@pytest.mark.parametrize('conv',[False,True])
def test_T4_slot_permutation_invariant(conv):
    net,site,x,y=pair(conv,True);opt=torch.optim.SGD(net.parameters(),lr=.01)
    expand(site,opt,candidates(site,x,y,nn.functional.mse_loss,1,torch.Generator().manual_seed(7)))
    with torch.no_grad():site.last.weight[:,3].fill_(.2)
    ref=output(net,x,False);order=[3,1,0,2];W=site.weights()
    rebuild(site,opt,order,W[columns(order,site.group_size)+[len(W)-1]])
    assert float((output(net,x,False)-ref).abs().max())<1e-12


@pytest.mark.parametrize('conv',[False,True])
def test_T5_compensation_matches_numpy_and_lstsq(conv):
    net,site,x,y=pair(conv,False)
    feature=torch.randn(12,3,6,6,dtype=torch.double) if conv else torch.randn(80,3,dtype=torch.double)
    A=design(site,feature);W=site.weights();G=A.T@A;S=columns([0,2],site.group_size)+[len(G)-1]
    actual=compensate(G,W,S);expected=torch.linalg.lstsq(A[:,S],A@W).solution
    ref=np.linalg.solve(G.numpy()[np.ix_(S,S)],G.numpy()[S]@W.numpy())
    assert float((actual-expected).norm()/expected.norm())<1e-8
    np.testing.assert_allclose(actual.numpy(),ref,rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize('r',[1,9])
def test_T6_exact_group_cost_matches_direct_residual(r):
    A=torch.randn(400,5*r,dtype=torch.double);A=torch.cat((A,torch.ones(400,1,dtype=torch.double)),1)
    W=torch.randn(5*r+1,3,dtype=torch.double);G=A.T@A;D=columns([1,3],r);S=[j for j in range(len(G)) if j not in D]
    res=float((A@W-A[:,S]@torch.linalg.lstsq(A[:,S],A@W).solution).square().sum())
    M=np.linalg.inv(G.numpy());ref=float(np.trace(W.numpy()[D].T@np.linalg.solve(M[np.ix_(D,D)],W.numpy()[D])))
    assert g_exact(G,W,D)==pytest.approx(res,rel=1e-8)
    assert g_exact(G,W,D)==pytest.approx(ref,rel=1e-8)


def test_T7_backward_downdate_matches_twenty_naive_lstsq_problems():
    rng=np.random.default_rng(0)
    for trial in range(20):
        r=1 if trial%2 else 9;m=5
        A=np.c_[rng.normal(size=(180,m*r)),np.ones(180)];W=rng.normal(size=(m*r+1,3));Z=A@W;G=A.T@A
        groups={u:list(range(u*r,(u+1)*r)) for u in range(m)}
        removed,Wc,alive=backward(torch.from_numpy(G),torch.from_numpy(W),groups,range(m),2)
        naive=[]
        for _ in range(2):
            def cost(u):
                D=[c for v in naive+[u] for c in groups[v]];S=[c for c in range(len(G)) if c not in D]
                return np.square(Z-A[:,S]@np.linalg.lstsq(A[:,S],Z,rcond=None)[0]).sum()
            naive.append(min([u for u in range(m) if u not in naive],key=cost))
        assert removed==naive
        ref=np.linalg.solve(G[np.ix_(alive,alive)],G[alive]@W)
        assert np.linalg.norm(Wc.numpy()-ref)/np.linalg.norm(ref)<1e-8


@pytest.mark.parametrize('kind',['sgd','adam'])
def test_T8_optimizer_and_bn_gather_bitwise(kind):
    net,site,x,y=pair(False,True)
    opt=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9) if kind=='sgd' else AdamRowStep(net.parameters())
    for _ in range(2):opt.zero_grad();nn.functional.mse_loss(net(x),y).backward();opt.step()
    expand(site,opt,candidates(site,x,y,nn.functional.mse_loss,1,torch.Generator().manual_seed(8)))
    if kind=='adam':
        assert bool((opt.state[site.first.weight]['step'][3]==0).all())
        assert bool((opt.state[site.last.weight]['step'][:,3]==0).all())
    opt.zero_grad();nn.functional.mse_loss(net(x),y).backward();opt.step()
    keep=[0,3,2];old={p:deepcopy(st) for p,st in opt.state.items()};oldfirst=site.first.weight;oldlast=site.last.weight;oldbn=site.bn.weight
    buffers={key:getattr(site.bn,key).clone() for key in ('running_mean','running_var')}
    rebuild(site,opt,keep)
    for original,new,axis in ((oldfirst,site.first.weight,0),(oldlast,site.last.weight,1),(oldbn,site.bn.weight,0)):
        for key,v in old[original].items():assert torch.equal(opt.state[new][key],v.index_select(axis,torch.tensor(keep)))
    for key,v in buffers.items():assert torch.equal(getattr(site.bn,key),v[keep])


def test_T9_disabled_matches_bp_bitwise_fifty_steps():
    net=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,1)).double();ref=deepcopy(net)
    a=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9);b=torch.optim.SGD(ref.parameters(),lr=.01,momentum=.9)
    engine=ETT(net,a,[LinearUnitLayer(net,'0','2','1')],rho=0.)
    for _ in range(50):
        x=torch.randn(4,2,dtype=torch.double);y=torch.randn(4,1,dtype=torch.double)
        engine.before_batch((x,y),nn.functional.mse_loss)
        for model,opt in ((net,a),(ref,b)):
            opt.zero_grad();nn.functional.mse_loss(model(x),y).backward();opt.step()
        engine.after_step((x,y))
    for p,q in zip(net.parameters(),ref.parameters()):assert torch.equal(p,q)
    for p,q in zip(net.parameters(),ref.parameters()):assert torch.equal(a.state[p]['momentum_buffer'],b.state[q]['momentum_buffer'])


@pytest.mark.parametrize('kind',['sgd','adam'])
def test_T10_zero_training_drop_virtual_is_bitwise_identity(kind):
    net,site,x,y=pair(False,True)
    opt=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9) if kind=='sgd' else AdamRowStep(net.parameters())
    opt.zero_grad();nn.functional.mse_loss(net(x),y).backward();opt.step()
    values=deepcopy(net.state_dict());state=deepcopy(opt.state_dict());ref=output(net,x,False)
    expand(site,opt,candidates(site,x,y,nn.functional.mse_loss,1,torch.Generator().manual_seed(5)))
    G=gram(site,[(x,y)]);W=site.weights();keep=[0,1,2];S=keep+[len(G)-1]
    out=compensate(G,W,S);rebuild(site,opt,keep,out)
    for key,v in values.items():assert torch.equal(net.state_dict()[key],v)
    now=opt.state_dict()
    for idx,st in state['state'].items():
        for key,v in st.items():assert torch.equal(now['state'][idx][key],v)
    assert torch.equal(output(net,x,False),ref)


def test_T11_trained_buffer_rolls_across_cycles_without_leakage():
    net=nn.Sequential(nn.Linear(1,2),nn.ReLU(),nn.Linear(2,1)).double()
    opt=torch.optim.SGD(net.parameters(),lr=.01)
    engine=ETT(net,opt,[LinearUnitLayer(net,'0','2','1')],T=10000,h=1)
    assert engine.buffer.capacity==1000
    for start in range(0,1200,100):
        x=torch.arange(start,start+100,dtype=torch.double)[:,None]/1500
        engine.after_step((x,x))
    assert engine.active is None
    stats,check,info=engine.buffer.split()
    import hashlib,json
    digest=lambda ids:hashlib.sha256(json.dumps(ids,separators=(',',':')).encode('ascii')).hexdigest()
    assert info['gram_ids']==[200,999] and info['check_ids']==[1000,1199]
    assert info['gram_ids_sha256']==digest(list(range(200,1000)))
    assert info['check_ids_sha256']==digest(list(range(1000,1200)))
    assert info['gram_ids'][1]<info['check_ids'][0]
    engine.begin(nn.functional.mse_loss)
    assert engine.active['init_buffer']['gram_ids']==info['gram_ids']
    x=torch.arange(1200,1500,dtype=torch.double)[:,None]/1500
    engine.after_step((x,x))
    stats,check,info=engine.buffer.split()
    assert info['gram_ids']==[500,1299] and info['check_ids']==[1300,1499]
    assert info['gram_ids_sha256']==digest(list(range(500,1300)))
    assert info['check_ids_sha256']==digest(list(range(1300,1500)))
    assert all(i<engine.buffer.seen for i in info['gram_ids']+info['check_ids'])
    saved=engine.state_dict();engine.load_state_dict(saved)
    assert engine.buffer.split()[2]==info
    records=engine.contract(nn.functional.mse_loss,acceptance=False)
    assert records[0]['gram_ids']==info['gram_ids'] and records[0]['check_ids']==info['check_ids']
    assert len(engine.buffer)==1000  # contraction must not discard pre-cycle history


def test_T12_out_of_sample_ls_error_is_bounded_with_large_buffer():
    torch.manual_seed(27)
    hidden=nn.Linear(10,6).double()
    X=torch.randn(1000,10,dtype=torch.double);future=torch.randn(1000,10,dtype=torch.double)
    with torch.no_grad():
        H=torch.relu(hidden(X));HF=torch.relu(hidden(future))
    A=torch.cat((H,torch.ones(1000,1,dtype=torch.double)),1)
    B=torch.cat((HF,torch.ones(1000,1,dtype=torch.double)),1)
    W=torch.randn(7,2,dtype=torch.double);S=[0,1,3,4,5,6]
    assert len(A)>=20*A.shape[1]
    trained=A[:800];fitted=compensate(trained.T@trained,W,S)
    in_sample=float((trained@W-trained[:,S]@fitted).square().sum()/len(trained))
    out_sample=float((B@W-B[:,S]@fitted).square().sum()/len(B))
    assert out_sample<=2*in_sample+1e-8
