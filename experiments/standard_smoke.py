"""Declared synthetic fixtures and a genuine PyCIL ResNet32 BN CPU correction."""
from contextlib import contextmanager
import json
import numpy as np
import torch


@contextmanager
def cifar_fixture(enabled):
    if not enabled:
        yield;return
    from utils import data_manager,inc_net
    from torchvision.transforms import ToTensor
    from convs.cifar_resnet import ResNetBasicblock
    from adapters.cil_setup import FixtureBackbone
    old_data,old_net=data_manager._get_idata,inc_net.get_convnet
    class DisjointCIFAR:
        use_path=False
        train_trsf=test_trsf=[]
        common_trsf=[ToTensor()]
        class_order=list(range(100))
        def download_data(self):
            generator=np.random.RandomState(914)
            self.train_data=generator.randint(256,size=(800,32,32,3),dtype=np.uint8)
            self.test_data=generator.randint(256,size=(200,32,32,3),dtype=np.uint8)
            self.train_targets=np.repeat(np.arange(100),8)
            self.test_targets=np.repeat(np.arange(100),2)
    data_manager._get_idata=lambda name:DisjointCIFAR()
    inc_net.get_convnet=lambda *a,**kw:FixtureBackbone(ResNetBasicblock)
    try:yield
    finally:data_manager._get_idata,inc_net.get_convnet=old_data,old_net


def resnet_bn_smoke():
    from experiments.runners.pycil import upstream_imports
    upstream_imports()
    from utils.inc_net import IncrementalNet
    from adapters.icarl_hooks import Prediction,sites_for
    from methods.cl_replay import ReplayLoss,ReplayCorrection,CorrectionConfig
    from methods.cl_state import State
    from diagnostics.vna_align import _global_state,_assert_bitwise
    with _global_state():
        torch.manual_seed(134)
        native=IncrementalNet({'convnet_type':'resnet32'},False);native.update_fc(5);native.train()
        x=torch.randn(2,3,32,32);y=torch.tensor([0,1]);model=Prediction(native)
        opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0005)
        losses=ReplayLoss(y,torch.tensor([False,True]))
        opt.zero_grad();losses.total(model(x)).backward();opt.step()
        engine=ReplayCorrection(CorrectionConfig(kernel='tiny_V',cg_tolerance=1e-4,cg_iterations=100,trace_probes=1,max_trials=3))
        before=State(model,opt)
        event=engine.step(model,opt,x,losses,sites_for(model)[-1:],.01,damping=10.,update_norm=.01)
        assert event['status'] in ('ACCEPTED','REJECTED'),event
        assert event['active_sites']==1,event
        for name,value in model.named_buffers():_assert_bitwise(value,before.buffers[name],name)
        return dict(backbone='PyCIL ResNet32 BN',synthetic=True,native_sgd_steps=1,correction=event)


if __name__=='__main__':
    torch.set_num_threads(1)
    print(json.dumps(resnet_bn_smoke(),indent=2))
