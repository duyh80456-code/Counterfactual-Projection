"""Independent H2 penalty; this is not the upstream LPR algorithm."""
import math
import time
import torch
from torch import nn
from methods.vna_v3 import training_products
from methods.cl_replay import ReplayCorrection
from methods.cl_state import predict,vector


class Features(nn.Module):
    def __init__(self,network,layers=('features',)):
        super().__init__();self.native=network;self.layers=tuple(layers)
        if not self.layers or len(set(self.layers))!=len(self.layers) or set(self.layers)-{'features','stage_1','stage_2','stage_3'}:
            raise ValueError('fixed feature layers: features, stage_1, stage_2, stage_3')
    def forward(self,x):
        result=self.native(x)
        return torch.cat([(result['features'] if name=='features' else result['fmaps'][int(name[-1])-1]).flatten(1)
                          for name in self.layers],dim=1)


class GeometryCorrection(ReplayCorrection):
    def step(self,model,optimizer,inputs,losses,sites,step_size,*,replay_inputs,geometry_weight,feature_layers=('features',),**kwargs):
        if inputs.is_cuda:torch.cuda.synchronize(inputs.device)
        started=time.perf_counter()
        if not math.isfinite(geometry_weight) or geometry_weight<0:raise ValueError('finite nonnegative feature penalty required')
        if replay_inputs is None or not len(replay_inputs):
            if geometry_weight:raise ValueError('positive H2 penalty needs replay features')
            event=super().step(model,optimizer,inputs,losses,sites,step_size,**kwargs)
            event.update(branch='H2',geometry_weight=0.,feature_layers=list(feature_layers),no_replay=True)
            return event
        native=model.native if hasattr(model,'native') else model
        features=Features(native,feature_layers)
        before=predict(features,replay_inputs).double();initial=vector(features)
        with training_products(features,[],{},replay_inputs) as products:
            # Freeze the feature linearization across the physical acceptance trials.
            products.parameter_refs=tuple((n,p.clone()) for n,p in products.parameter_refs)
            train_before=products._call({}).detach().double()
            penalty=lambda d:geometry_weight*products.JT(products.Jv(d)).double()
            event=super().step(model,optimizer,inputs,losses,sites,step_size,penalty=penalty,**kwargs)
            predicted=products.Jv(vector(features)-initial).double()
            event.update(predicted_train_feature_drift=float(predicted.norm()),feature_jvp=products.jvp_calls,feature_vjp=products.vjp_calls)
        event.update(branch='H2',geometry_weight=geometry_weight,metric='JtHJ+lambda*Jphi_t_Jphi+mu*I',
            cotangent_recomputed=True,implementation='VNA feature penalty; not LPR',feature_layers=list(feature_layers),
            penalty_form='fixed lambda; not an inferred constrained-problem KKT multiplier')
        after=predict(features,replay_inputs).double()
        actual=(predict(features,replay_inputs,'train').double().flatten()-train_before)
        event.update(replay_feature_drift=float((after-before).norm()),
            feature_cosine=float(torch.nn.functional.cosine_similarity(before.flatten(1),after.flatten(1),dim=1).mean()),
            feature_measurement_mode='eval; distinct from train-mode penalty',train_feature_drift=float(actual.norm()),
            feature_linearization_error=float((actual-predicted).norm()),correction_core_seconds=event['seconds'],
            seconds=time.perf_counter()-started)
        if inputs.is_cuda:torch.cuda.synchronize(inputs.device)
        event.update(jvp=event.get('jvp',0)+event['feature_jvp'],vjp=event.get('vjp',0)+event['feature_vjp'],
            seconds=time.perf_counter()-started,
            peak_vram_bytes=torch.cuda.max_memory_allocated(inputs.device) if inputs.is_cuda else 0)
        return event
