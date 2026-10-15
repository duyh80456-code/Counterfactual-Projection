"""Reproduce with PYTHONPATH=. PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python .venv/bin/python docs/vna/measure_efficiency_logs/fd_numerical_audit.py.
The probe uses the same float64 site fixtures/directions and reports step-size sensitivity.
"""
import torch
from adapters.repdistiller_hooks import upstream, sites_for
from experiments.stage_b4 import build_model, sites_for as b4_sites
from methods.vna_joint import joint_products
from diagnostics.vna_align import _global_state

torch.set_num_threads(1)
def audit(model, sites, size, label):
    model.eval()
    for i, site in enumerate(sites):
        x=torch.randn(1,3,size,size,dtype=torch.float64)
        A={'weight':site.first_layer.layer.weight[:1].detach().clone(),'bias':None}
        with _global_state(), joint_products(model,[site],{site:A},x) as p:
            expected=model(x).flatten()
            w=torch.randn(sum(t.numel() for _,t in p.extension_refs),dtype=x.dtype)
            u=torch.randn(expected.numel(),dtype=x.dtype)
            j=p.JEv(w); adj=abs(float(u@j-w@p.JET(u)))
            pieces=w.split([t.numel() for _,t in p.extension_refs])
            for eps in (1e-5,1e-6,1e-7,1e-8):
                masks=[]
                handles=[m.register_forward_hook(lambda m,a,o: masks.append((o.detach()>0).clone())) for m in model.modules() if isinstance(m,torch.nn.ReLU)]
                plus=p._call({n:t+eps*v.reshape_as(t) for (n,t),v in zip(p.extension_refs,pieces)})
                split=len(masks)
                minus=p._call({n:t-eps*v.reshape_as(t) for (n,t),v in zip(p.extension_refs,pieces)})
                for h in handles:h.remove()
                flips=sum(int((a!=b).sum()) for a,b in zip(masks[:split],masks[split:]))
                fd=(plus-minus)/(2*eps)
                tight=torch.allclose(j,fd,rtol=1e-5,atol=1e-7)
                print(label,i,eps,'rel_l2',float((fd-j).norm()/j.norm()) if j.norm() else 0.,'max_abs',float((fd-j).abs().max()),'relu_flips',flips,'adj',adj,'passes',tight,flush=True)
for seed in (0,1):
    torch.manual_seed(seed)
    with upstream() as models:
        model=models['wrn_16_2'](num_classes=100).double(); sites=sites_for(model)
        audit(model,sites[::2],32,f'WRN seed={seed}')
    torch.manual_seed(seed)
    model=build_model().double(); audit(model,b4_sites(model),8,f'B4 seed={seed}')
