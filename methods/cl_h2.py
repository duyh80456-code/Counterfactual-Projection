"""Independent H2 penalty; this is not the upstream LPR algorithm."""
import math
import torch
from torch import nn
from methods.vna_v3 import training_products
from methods.cl_replay import ReplayCorrection
from methods.cl_state import predict


class Features(nn.Module):
    def __init__(self,network):super().__init__();self.native=network
    def forward(self,x):return self.native(x)['features']


class GeometryCorrection(ReplayCorrection):
    def step(self,model,optimizer,inputs,losses,sites,step_size,*,replay_inputs,geometry_weight):
        if not math.isfinite(geometry_weight) or geometry_weight<0:raise ValueError('finite nonnegative feature penalty required')
        if replay_inputs is None or not len(replay_inputs):
            if geometry_weight:raise ValueError('positive H2 penalty needs replay features')
            return super().step(model,optimizer,inputs,losses,sites,step_size)
        native=model.native if hasattr(model,'native') else model
        before=predict(Features(native),replay_inputs).double()
        with training_products(Features(native),[],{},replay_inputs) as products:
            penalty=lambda d:geometry_weight*products.JT(products.Jv(d)).double()
            event=super().step(model,optimizer,inputs,losses,sites,step_size,penalty=penalty)
        event.update(branch='H2',geometry_weight=geometry_weight,metric='JtHJ+lambda*Jphi_t_Jphi+mu*I',
            cotangent_recomputed=True,implementation='VNA feature penalty; not LPR')
        after=predict(Features(native),replay_inputs).double()
        event.update(replay_feature_drift=float((after-before).norm()),
            feature_cosine=float(torch.nn.functional.cosine_similarity(before.flatten(1),after.flatten(1),dim=1).mean()),
            feature_measurement_mode='eval; distinct from train-mode penalty')
        return event
