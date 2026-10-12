"""Interchangeable temporary E generators, frozen O; SSD intentionally not registered."""
from dataclasses import dataclass, replace
import time
import torch
import torch.nn.functional as F
from adapters.deit_mlp_growth import DeitMLPGrowthAdapter, MLPExpansionCandidate
from diagnostics.deit_expansion_proposals import frozen_original, gradient_tuned
from diagnostics.deit_expansion_suite import private_rng

@dataclass(frozen=True)
class GeneratorConfig:
    rank: int=4
    seed: int=5
    optimized_steps: int=20
    gradmax_steps: int=5
    learning_rate: float=.001

    def __post_init__(self):
        if self.rank<1 or not 1<=self.optimized_steps<=20 or not 1<=self.gradmax_steps<=20 or self.learning_rate<=0:
            raise ValueError('invalid generator budget')

@dataclass
class ExpansionCandidate:
    branch: MLPExpansionCandidate
    generator: str
    gate: float
    budget: dict

    @property
    def site(self):return self.branch.module_name

    @torch.no_grad()
    def delta_logits(self,batch):
        model=self.branch.model
        base=model(batch[0]).detach()
        with self.branch.virtual_direction(self.gate):expanded=model(batch[0]).detach()
        return expanded-base

class ExpansionGenerator:
    def generate(self,model,site,statistics_batches,tuning_batches,config):raise NotImplementedError

class TinyGenerator(ExpansionGenerator):
    def generate(self,model,site,statistics_batches,tuning_batches,config):
        started=time.perf_counter()
        branch=DeitMLPGrowthAdapter().propose_auxiliary_growth(model=model,site=site,batches=statistics_batches,rank=config.rank)
        return ExpansionCandidate(branch,'tiny',.05,{'forward_calls':len(statistics_batches),'backward_calls':len(statistics_batches),
            'search_time_seconds':time.perf_counter()-started,'source':'native_TINY','seed':config.seed})

class OptimizedEGenerator(ExpansionGenerator):
    def generate(self,model,site,statistics_batches,tuning_batches,config):
        started=time.perf_counter()
        branch,record=gradient_tuned(model,site,tuning_batches[0],rank=config.rank,steps=config.optimized_steps,
            learning_rate=config.learning_rate,seed=config.seed)
        return ExpansionCandidate(branch,'optimized_e',1.,{'forward_calls':record['proposal_forward_calls'],
            'backward_calls':record['proposal_backward_calls'],'search_time_seconds':time.perf_counter()-started,
            'source':'full_A_a_B_Adam_frozen_O_not_B_only_GN','seed':config.seed,'tuning':record})

class GradMaxGenerator(ExpansionGenerator):
    """Incoming zero, outgoing SVD initialization, then incoming-only short steps.

    GELU(0)=0 and GELU'(0)=1/2 meet the initialization assumptions. This is an
    auxiliary-only adaptation, not the paper's full growing-network training run.
    """
    def generate(self,model,site,statistics_batches,tuning_batches,config):
        started=time.perf_counter();mlp=DeitMLPGrowthAdapter.resolve_site(model,site);captures={}
        matrix=mlp.fc2.weight.new_zeros(mlp.fc2.out_features,mlp.fc1.in_features+1);samples=0
        with frozen_original(model):
            def before(_module,inputs):captures['z']=inputs[0].detach()
            def after(_module,inputs,output):
                leaf=output.detach().requires_grad_();captures['leaf']=leaf;return leaf
            pre=mlp.register_forward_pre_hook(before);post=mlp.register_forward_hook(after)
            try:
                for inputs,labels in statistics_batches:
                    logits=model(inputs);loss=F.cross_entropy(logits.float(),labels,reduction='sum')
                    gradient=torch.autograd.grad(loss,captures['leaf'])[0].detach()
                    z=captures['z'].reshape(-1,mlp.fc1.in_features);g=gradient.reshape(-1,mlp.fc2.out_features)
                    augmented=torch.cat((z,z.new_ones(len(z),1)),1)
                    matrix.add_(g.T@augmented);samples+=len(labels)
            finally:pre.remove();post.remove()
            matrix.mul_(.5/samples) # GELU derivative at zero, mean over images.
            U,s,_=torch.linalg.svd(matrix.double(),full_matrices=False)
            rank=min(config.rank,U.shape[1]);B=U[:,:rank].to(mlp.fc2.weight)/rank**.5
            A=mlp.fc1.weight.new_zeros(rank,mlp.fc1.in_features).requires_grad_()
            a=mlp.fc1.weight.new_zeros(rank).requires_grad_()
            branch=MLPExpansionCandidate(site,model,A,a,B,0.,{'source':'GradMax_GELU_SVD','effective_rank':rank})
            opt=torch.optim.SGD([A,a],lr=config.learning_rate);losses=[]
            for _ in range(config.gradmax_steps):
                opt.zero_grad(set_to_none=True)
                with branch.virtual_direction(1.):loss=F.cross_entropy(model(tuning_batches[0][0]).float(),tuning_batches[0][1])
                if not torch.isfinite(loss):raise ValueError('nonfinite GradMax auxiliary loss')
                losses.append(float(loss.detach()));loss.backward();opt.step()
        branch=replace(branch,A=A.detach().clone(),a=a.detach().clone(),B=B.detach().clone())
        return ExpansionCandidate(branch,'gradmax',1.,{'forward_calls':len(statistics_batches)+config.gradmax_steps,
            'backward_calls':len(statistics_batches)+config.gradmax_steps,'search_time_seconds':time.perf_counter()-started,
            'source':'GradMax_GELU_initialization_plus_incoming_only_steps','seed':config.seed,
            'initialization':'A_a_zero_B_left_singular_vectors_Frobenius_norm1',
            'outgoing_constraint':'rank_k_orthogonal_columns_BtB_equals_I_over_rank',
            'singular_values':s.detach().cpu().tolist(),'initialization_degenerate':float(s.max())<=1e-12,
            'auxiliary_train_scope':'A_a_only_B_fixed_O_frozen','tuning_losses':losses,
            'paper':'https://arxiv.org/abs/2201.05125'})

GENERATORS={'tiny':TinyGenerator,'optimized_e':OptimizedEGenerator,'gradmax':GradMaxGenerator}


def generate(model,site,statistics,tuning,config,name):
    with private_rng(config.seed):return GENERATORS[name]().generate(model,site,statistics,tuning,config)
