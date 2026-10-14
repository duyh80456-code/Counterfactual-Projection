"""Stage 2 dense independent oracles, state preservation and selection tests."""
import copy
from dataclasses import replace
import pytest
import torch
from experiments.vna_support import tiny_model, tiny_candidate, isolated_rng
from methods.vna import block_products, extract_tiny_A, dedicated_generator, _unflatten
from methods.vna_ggn import (GGNProducts, GGNConfig, vna_exact_ggn, vna_lite_ggn,
    ngd_control, project_bn_tangent, bn_conv_weights, filtered_cotangent)
from methods.vna_kfac import KFACBlockPreconditioner, vna_kfac
from methods.vna_select import tiny_V, greedy_V, trace_matched_random_A, extension_trace, neuron_norms


@pytest.fixture
def toy():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    gen = dedicated_generator(77, 'stage2-toy')
    with isolated_rng(gen):
        model = tiny_model()
    batch = (torch.randn(3, 3, 8, 8, generator=gen), torch.tensor([0, 1, 2]))
    site = model.growing_blocks()[0].name
    a, _ = extract_tiny_A(model, tiny_candidate(model, site, batch, 2))
    yield model, site, batch, a
    torch.set_num_threads(threads)


def flat(delta, refs):
    return torch.cat([delta[n].flatten() for n, _ in refs]).double()


def dense(p, targets):
    op = GGNProducts(p, targets)
    eye = torch.eye(op.g.numel(), dtype=p.inputs.dtype)
    J = torch.stack([p.JT(e).double() for e in eye])
    E = torch.stack([p.JET(e).double() for e in eye])
    H = torch.stack([op.H(e) for e in eye], 1)
    return op, J, E, H


def state(m):
    return ({n: t.clone() for n, t in m.state_dict().items()},
            {n: None if p.grad is None else p.grad.clone() for n, p in m.named_parameters()},
            [x.training for x in m.modules()], torch.get_rng_state().clone())


def unchanged(m, before):
    after = state(m)
    for n, t in before[0].items():
        assert torch.equal(t, after[0][n])
    for n, t in before[1].items():
        assert (after[1][n] is None) if t is None else torch.equal(t, after[1][n])
    assert before[2] == after[2]
    assert torch.equal(before[3], after[3])


def test_dense_solvers_function_equivalence_and_descent(toy, monkeypatch):
    m, s, b, a = toy
    m = copy.deepcopy(m).double()
    b = (b[0].double(), b[1]); a = {n: None if v is None else v.double() for n, v in a.items()}
    cfg = GGNConfig(project_bn=False, cg_tol=1e-9, cg_iters=500)
    with block_products(m, s, a, b[0]) as p:
        op, J, E, H = dense(p, b[1]); refs = p.parameter_refs
        F = J.T @ H @ J; C = J.T @ H @ E
        mu = .01 * float(torch.trace(F)) / F.shape[0]
        R = torch.linalg.inv(F + mu * torch.eye(len(F), dtype=F.dtype))
        grad = J.T @ op.g; exact = R @ C @ C.T @ R @ grad
        ngd = R @ F @ R @ grad
        K, KE = J @ J.T, E @ E.T
        Q = torch.linalg.inv(H @ K + mu * torch.eye(len(H), dtype=H.dtype))
        function = J.T @ Q @ H @ KE @ H @ K @ Q @ op.g
        assert torch.allclose(exact, function, rtol=2e-7, atol=1e-10)
        lite = J.T @ H @ KE @ H @ K @ op.g / (torch.trace(F)/len(H))**2
    for fn, oracle in ((vna_exact_ggn, exact), (ngd_control, ngd), (vna_lite_ggn, lite)):
        d, diag = fn(m, s, a, b, cfg, mu=mu, trace_F=float(torch.trace(F)))
        assert torch.allclose(flat(d, refs), oracle, rtol=2e-6, atol=1e-9)
        assert diag['raw_gain'] >= -1e-12
        assert all(r['converged'] for r in diag['cg'])
    z = filtered_cotangent(m,s,a,b,cfg,mu=mu).flatten()
    assert torch.allclose(z, H @ J @ R @ grad, rtol=1e-7, atol=1e-10)
    def no_cg(*args, **kwargs):
        raise AssertionError('lite must not solve a linear system')
    monkeypatch.setattr('methods.vna_ggn.solve_ggn', no_cg)
    d,diag=vna_lite_ggn(m,s,a,b,cfg,mu=mu,trace_F=float(torch.trace(F)))
    assert torch.allclose(flat(d,refs),lite,rtol=2e-6,atol=1e-9)
    assert diag['cg']==[]


def test_weighted_products_finite_difference_and_adjoint(toy):
    m,s,b,a = toy
    m=m.double(); b=(b[0].double(),b[1]); a={n:None if v is None else v.double() for n,v in a.items()}
    with block_products(m,s,a,b[0]) as p:
        op,J,E,H=dense(p,b[1])
        gen=dedicated_generator(2,'weighted-fd')
        u=torch.randn(J.shape[1],dtype=torch.float64,generator=gen); u/=u.norm()
        v=torch.randn(E.shape[1],dtype=torch.float64,generator=gen); v/=v.norm()
        assert torch.allclose(op.F(u),J.T@H@J@u,atol=1e-11)
        assert torch.allclose(op.COE(v),J.T@H@E@v,atol=1e-11)
        assert torch.allclose(op.CEO(u),E.T@H@J@u,atol=1e-11)
        assert torch.allclose(u.dot(op.COE(v)),v.dot(op.CEO(u)),atol=1e-11)
        for refs,direction,expected in ((p.parameter_refs,u,J@u),(p.extension_refs,v,E@v)):
            eps=1e-5
            delta=_unflatten(direction,refs)
            plus=p._call({n:r+eps*delta[n] for n,r in refs})
            minus=p._call({n:r-eps*delta[n] for n,r in refs})
            assert torch.allclose(op.H((plus-minus)/(2*eps)),H@expected,rtol=1e-6,atol=1e-10)


def test_invariance_projection_and_norm_control(toy):
    m,s,b,a=toy
    m.train(); next(m.parameters()).grad=torch.ones_like(next(m.parameters()))
    before=state(m)
    k=KFACBlockPreconditioner(); k.refresh(m,s,b)
    for fn in (vna_exact_ggn,vna_lite_ggn,ngd_control):
        d,_=fn(m,s,a,b)
        unchanged(m,before)
    d,_=vna_kfac(m,s,a,b,k); unchanged(m,before)
    projected=project_bn_tangent(m,s,d)
    for n,w in bn_conv_weights(m,s).items():
        assert abs(float((projected[n].double()*w.double()).sum())) < 1e-10
    for n in d:
        if n not in bn_conv_weights(m,s):
            assert torch.equal(d[n],projected[n])
    control,diag=ngd_control(m,s,a,b,reference=d)
    assert sum(float(t.square().sum()) for t in control.values()) == pytest.approx(
        sum(float(t.square().sum()) for t in d.values()),rel=1e-10)


@pytest.mark.parametrize('corrected',[False,True])
def test_kfac_dense_inverse_psd_state_and_exact_diagonal(toy,corrected):
    m,s,b,a=toy
    k=KFACBlockPreconditioner(ema=.3,eigenvalue_correction=corrected); k.refresh(m,s,b)
    with block_products(m,s,a,b[0]) as p:
        op,J,E,H=dense(p,b[1]); refs=p.parameter_refs
        x=torch.linspace(-1,1,op.size,dtype=torch.float64)
        mx=k.apply(x,refs,.01)
        y=x.flip(0); my=k.apply(y,refs,.01)
        assert x.dot(mx)>0
        assert x.dot(my)==pytest.approx(float(y.dot(mx)),rel=1e-10,abs=1e-10)
        for prefix,block in k.blocks.items():
            n=prefix+'.weight'; idx=[n2 for n2,_ in refs].index(n)
            start=sum(r.numel() for _,r in refs[:idx]); size=refs[idx][1].numel()
            if not corrected:
                F=torch.kron(block['G'].contiguous(),block['A'].contiguous())
                expected=torch.linalg.solve(F+.01*torch.eye(size,dtype=torch.float64),x[start:start+size])
                assert torch.allclose(mx[start:start+size],expected,rtol=1e-9,atol=1e-9)
            else:
                basis=torch.kron(block['QG'].contiguous(),block['QA'].contiguous())
                exact=J[:,start:start+size].T@H@J[:,start:start+size]
                assert torch.allclose(block['eigenvalues'].flatten(),torch.diag(basis.T@exact@basis),rtol=1e-5,atol=1e-10)
        c=KFACBlockPreconditioner(); c.load_state_dict(k.state_dict())
        assert torch.equal(mx,c.apply(x,refs,.01))
    gen=dedicated_generator(44,'kfac-random-batches')
    for _ in range(5):
        random_batch=(torch.randn(b[0].shape,generator=gen),torch.randint(3,b[1].shape,generator=gen))
        _,diag=vna_kfac(m,s,a,random_batch,k,GGNConfig(project_bn=False))
        assert diag['raw_gain'] >= -1e-10
    k.refresh(m,s,b)
    assert k.refreshes==2


def test_tiny_pseudo_loss_parity_generalized_eigen_and_invariance(toy):
    m,s,b,a=toy
    m.eval()
    with torch.no_grad():
        logits=m(b[0]); z=logits.softmax(-1)-torch.nn.functional.one_hot(b[1],3)
    capture={}
    candidate=tiny_candidate(m,s,b,2,capture=capture)
    expected,_=extract_tiny_A(m,candidate)
    before=state(m)
    actual,stats=tiny_V(m,s,[b],2,cotangents=[z],existing_weight_projection=True,normalize=False,score_signs=False)
    unchanged(m,before)
    assert torch.allclose(actual['weight'],expected['weight'],rtol=1e-5,atol=1e-7)
    assert torch.allclose(stats['N_z'],capture['N'],rtol=1e-5,atol=1e-8)
    # The requested z:=g test: mean cotangent reproduces A directions;
    # raw upstream A carries sqrt(singular value), hence sqrt(batch) scaling.
    mean_A,_=tiny_V(m,s,[b],2,cotangents=[z/len(z)],existing_weight_projection=True,
                   normalize=False,score_signs=False)
    assert torch.allclose(mean_A['weight']*len(z)**.5,expected['weight'],rtol=2e-4,atol=1e-7)
    assert torch.allclose(stats['generalized_eigenvalues'][:2],stats['singular_values'].square(),rtol=2e-4,atol=1e-9)
    multi,_=tiny_V(m,s,[b,b],2,cotangents=[z,z],existing_weight_projection=True,normalize=False,score_signs=False)
    assert torch.allclose(multi['weight'],actual['weight'],rtol=1e-4,atol=1e-7)


def test_greedy_gain_no_duplicates_norm_and_trace_control(toy):
    m,s,b,a=toy
    z=filtered_cotangent(m,s,a,b)
    before=state(m)
    av,_=tiny_V(m,s,[b],2,cotangents=[z])
    ag,diag=greedy_V(m,s,av,b[0],z,random_count=2,r_max=3)
    assert len(diag['selected'])==len(set(diag['selected']))
    assert all(y>=x-1e-10 for x,y in zip(diag['gains'],diag['gains'][1:]))
    assert diag['threshold_reached']
    mean=m.block(s).first_layer.layer.weight.detach().flatten(1).norm(dim=1).mean()
    assert torch.allclose(neuron_norms(ag),mean.expand(len(ag['weight'])),rtol=1e-5)
    ar,_=trace_matched_random_A(m,s,av,b[0])
    assert extension_trace(m,s,ar,b[0])==pytest.approx(extension_trace(m,s,av,b[0]),rel=1e-5)
    unchanged(m,before)


def test_validation_cli_artifacts_equal_rank_and_cross_ablation(tmp_path):
    from diagnostics.vna_solver_validation import run
    old_threads=torch.get_num_threads()
    try:
        summary=run(tmp_path)
    finally:
        torch.set_num_threads(old_threads)
    assert len(summary['rows'])==24
    assert len(summary['selection'])==16
    assert all(r['cos_W1']==0 and r['norm_W1']==0 for r in summary['rows']
               if r['variant']=='cross_dropped')
    assert all(r['cg_converged'] for r in summary['rows'] if r['variant']=='exact')
    for relative in (1e-4,1e-3,1e-2,1e-1):
        assert len({r['rank'] for r in summary['selection'] if r['relative_mu']==relative})==1
    assert (tmp_path/'solvers.csv').is_file()
    assert (tmp_path/'selection.csv').is_file()
    assert (tmp_path/'summary.json').is_file()
