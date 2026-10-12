"""Projection mathematics, precision-compatible generators and immutable previews."""
from dataclasses import asdict,replace
import copy
import torch
import pytest
from torch import nn
from projection.functional import FunctionalProjector
from projection.exact_reference import svd_reference,solver_error
from projection.margin_diagnostics import margin_transfer
from adapters.deit_cp_adapter import CPConfig,ResearchCPConfig,make_projector
from adapters.deit_expansion_generators import GeneratorConfig,GradMaxGenerator,OptimizedEGenerator
from adapters.deit_projection_aware import tensor_hash
from diagnostics.deit_projection_v2 import analyze


def test_relative_trace_cg_matches_svd_and_absolute_legacy():
    model=nn.Linear(3,3,bias=False).double()
    x=torch.tensor([[1.,0.,0.]],dtype=torch.float64);target=torch.tensor([[1.,2.,3.]],dtype=torch.float64)
    # Each output has one unit derivative, hence trace(JJ^T)/m=1 exactly.
    relative=FunctionalProjector(.1,20,tolerance=1e-12,max_damping_retries=0,
        preconditioner_probes=0,damping_mode='relative_gram',trace_probes=4).project(model,x,target,block='')
    old=FunctionalProjector(.1,20,tolerance=1e-12,max_damping_retries=0,preconditioner_probes=0).project(model,x,target,block='')
    explicit=FunctionalProjector(.1,20,tolerance=1e-12,max_damping_retries=0,
        preconditioner_probes=0,damping_mode='absolute').project(model,x,target,block='')
    assert relative.gram_trace_mean==pytest.approx(1.) and relative.damping_used==pytest.approx(.1)
    assert torch.equal(old.fitted_delta,explicit.fitted_delta)
    assert torch.allclose(relative.fitted_delta,target/1.1,atol=1e-10)
    J=torch.zeros(3,9,dtype=torch.float64)
    for i in range(3):J[i,i*3]=1.
    ref=svd_reference(J,target,.1)
    delta=relative.parameter_delta['weight'].reshape(-1)
    assert solver_error(J,target,delta,ref)['solver_functional_error']<1e-10
    # Scaling the Jacobian by c scales mu by c^2 and leaves fitted target fraction.
    scaled=FunctionalProjector(.1,20,tolerance=1e-12,max_damping_retries=0,
        preconditioner_probes=0,damping_mode='relative_gram',trace_probes=4).project(model,3*x,target,block='')
    assert scaled.gram_trace_mean==pytest.approx(9.)
    assert torch.allclose(scaled.fitted_delta,relative.fitted_delta,atol=1e-10)
    assert 'damping_mode' not in asdict(CPConfig()) # no old checkpoint schema change
    assert make_projector(ResearchCPConfig(damping_mode='relative_gram')).damping_mode=='relative_gram'


def test_svd_separates_range_regularization_and_solver_error():
    J=torch.diag(torch.tensor([1.,2.,0.],dtype=torch.float64));d=torch.ones(3,dtype=torch.float64)
    ref=svd_reference(J,d,.5)
    assert ref['minimum_residual']==pytest.approx(1/3**.5)
    assert ref['ridge_residual']**2==pytest.approx(ref['minimum_residual']**2+ref['regularization_error']**2)
    assert solver_error(J,d,ref['ridge_delta'],ref)['solver_functional_error']==0
    assert solver_error(J,d,torch.zeros(3),ref)['solver_functional_error']>0
    with pytest.raises(ValueError,match='small matrices'):svd_reference(torch.zeros(100,100),torch.ones(100),.1,max_elements=10)


def test_margin_bound_and_nonlinear_diagnostics():
    base=torch.tensor([[0.,1.],[2.,0.]],dtype=torch.float64);labels=torch.tensor([0,0])
    expanded=torch.tensor([[3.,0.],[3.,0.]],dtype=torch.float64);linear=expanded-base
    actual=expanded+torch.tensor([[.1,-.1],[-.1,.1]],dtype=torch.float64)
    report=margin_transfer(base,expanded,linear,actual,labels)
    assert report['bound_violations']==0 and report['sufficient_bound_samples']==2
    assert report['corrected_samples']==1 and report['damaged_samples']==0
    assert report['nonlinear_error_linf_mean']==pytest.approx(.1)


def test_gradmax_and_full_optimized_E_do_not_modify_O(deit_small,deit_batches):
    model=deit_small.eval();before=tensor_hash(model.state_dict());ids={name:id(p) for name,p in model.named_parameters()}
    config=GeneratorConfig(gradmax_steps=2,optimized_steps=2,learning_rate=.01)
    for generator in (GradMaxGenerator(),OptimizedEGenerator()):
        candidate=generator.generate(model,'blocks.0.mlp',deit_batches[:2],[deit_batches[3]],config)
        assert torch.isfinite(candidate.delta_logits(deit_batches[0])).all()
        assert candidate.budget['backward_calls']>0
        assert tensor_hash(model.state_dict())==before
        assert ids=={name:id(p) for name,p in model.named_parameters()}
        assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    candidate=GradMaxGenerator().generate(model,'blocks.0.mlp',deit_batches[:2],[deit_batches[3]],config)
    batches={'projection_batch':deit_batches[0],'gate_batch':deit_batches[1],
        'heldout':deit_batches[2],'where_batches':deit_batches[1:]}
    cp=ResearchCPConfig(cg_iterations=3,cg_preconditioner_probes=0,damping_mode='relative_gram',trace_probes=2)
    _,record=analyze(model,candidate,batches,cp)
    assert record['heldout_role'].startswith('independent')
    assert record['heldout_margin_diagnostics']['bound_violations']==0
    assert not record['correction_applied'] and not record['range_limit_known']
    assert tensor_hash(model.state_dict())==before


def test_generator_initializations_preserve_function_at_gate_one(deit_small,deit_batches):
    from adapters.deit_mlp_growth import MLPExpansionCandidate
    from adapters.deit_expansion_generators import ExpansionCandidate
    model=deit_small.eval();mlp=model.blocks[0].mlp
    A=torch.randn(2,mlp.fc1.in_features,dtype=mlp.fc1.weight.dtype)
    a=torch.randn(2,dtype=A.dtype);B=torch.randn(mlp.fc2.out_features,2,dtype=A.dtype)
    # GradMax zero incoming; Optimized-E zero outgoing. Both are zero-function
    # expansions at gate1, stronger than the generic gate0 preservation test.
    for incoming,bias,outgoing in ((torch.zeros_like(A),torch.zeros_like(a),B),(A,a,torch.zeros_like(B))):
        branch=MLPExpansionCandidate('blocks.0.mlp',model,incoming,bias,outgoing,0.,{})
        candidate=ExpansionCandidate(branch,'initialization_test',1.,{})
        assert float(candidate.delta_logits(deit_batches[0]).abs().max())<1e-6
    assert not any(module._forward_hooks for module in model.modules())


def test_experimental_selector_uses_gate_not_E_or_heldout():
    from diagnostics.deit_projection_v2 import choose_projected_candidate
    rows=[{'gate_accepted':True,'projection_scope':['blocks.0.mlp'],'gate_gain':.1,
        'gate_margin_gain_after_projection':.01,'e_gain':99.,'heldout_cosine':1.},
        {'gate_accepted':True,'projection_scope':['blocks.1.mlp'],'gate_gain':.02,
        'gate_margin_gain_after_projection':.2,'e_gain':0.,'heldout_cosine':0.}]
    assert choose_projected_candidate(rows) is rows[0]
    assert choose_projected_candidate(rows,'gate_margin_gain_after_projection') is rows[1]
    rows[1]['gate_accepted']=False
    assert choose_projected_candidate(rows,'gate_margin_gain_after_projection') is rows[0]


def test_research_resume_preserves_original_reference_and_latest_bytes(tmp_path):
    from experiments.deit_research_resume import import_research
    from experiments.shared_protocol import sha256_file
    mounted=tmp_path/'mounted';mounted.mkdir();output=tmp_path/'output'
    protocol={'version':3,'seed':5}
    reference=mounted/'renamed_reference.pt'
    torch.save({'kind':'deit_v2_generator_reference','run_identity':{
        'protocol':protocol,'raw_reference_hash':'raw5'},'candidates':{'tiny':torch.ones(2)}},reference)
    method='cp_v2_tiny_relative_gram';latest=mounted/'renamed_progress.pt'
    payload={'kind':'deit_projection_aware_latest','protocol':protocol,
        'run_identity':{'method':method,'generator_reference_hash':sha256_file(reference)},
        'completed_epochs':12,'model':{'tensor':torch.ones(2)},'optimizer':{'moments':torch.ones(2)}}
    torch.save(payload,latest)
    import_research([mounted],output,{'protocol':protocol},'raw5')
    assert sha256_file(output/'generator_reference.pt')==sha256_file(reference)
    assert sha256_file(output/method/'checkpoint_latest.pt')==sha256_file(latest)
    payload['run_identity']['generator_reference_hash']='wrong'
    torch.save(payload,latest)
    with pytest.raises(ValueError,match='another generator reference'):
        import_research([mounted],tmp_path/'bad',{'protocol':protocol},'raw5')


def test_seed5_notebook_v2_stages_and_code_syntax():
    import ast,json
    from pathlib import Path
    notebook=json.loads(Path('notebooks/kaggle_deit_tiny_seed5_raw_vs_vanilla_rollback.ipynb').read_text())
    code='\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code')
    assert 'RUN_CONTINUATIONS = False' in code and "['--prepare-only']" in code
    assert "'all-continuations'" in code and "'--input-root', '/kaggle/input'" in code
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))


def test_heldout_appends_without_changing_original_partition(monkeypatch):
    import experiments.run_deit_research_v2 as runner
    from types import SimpleNamespace
    from torch.utils.data import TensorDataset
    ids={'statistics':[0,1],'where':[[2,3]],'projection':[4,5],'gate':[6,7],
         'opt_fit':[8,9],'opt_val':[10,11]}
    original=copy.deepcopy(ids)
    monkeypatch.setattr(runner,'materialize_probe_batches',lambda *_a,**_kw:({},copy.deepcopy(ids)))
    data=TensorDataset(torch.arange(400).reshape(400,1),torch.zeros(400,dtype=torch.long))
    batches,result=runner.research_batches(data,list(range(400)),SimpleNamespace(seed=5),None,torch.device('cpu'))
    assert {key:result[key] for key in original}==original
    assert len(result['heldout'])==256 and len(batches['heldout'][1])==256
    assert not set(result['heldout']).intersection(range(12))
    assert batches['heldout'][0].flatten().tolist()==result['heldout']
