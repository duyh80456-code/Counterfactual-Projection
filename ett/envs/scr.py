"""Original slowly-changing regression generator (upstream data code only)."""
from pathlib import Path
import sys
import torch


UPSTREAM=Path(__file__).resolve().parents[2]/'third_party/loss-of-plasticity'


def data(cfg,root):
    path=Path(root)/'scr.pt';path.parent.mkdir(parents=True,exist_ok=True)
    identity={k:cfg[k] for k in ('data_seed','steps','flip_after','inputs','target_features','flipping_bits','beta')}
    if path.exists():
        saved=torch.load(path,weights_only=False)
        if saved['identity']!=identity:raise ValueError('SCR cached data identity mismatch')
        return saved['x'],saved['y']
    sys.path.insert(0,str(UPSTREAM))
    from lop.nets.fix_ltu_net import FixLTUNet
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['data_seed'])
        target=FixLTUNet(cfg['inputs'],cfg['target_features'],cfg['beta'])
        count=cfg['steps']+cfg['flip_after'];tasks=count//cfg['flip_after']+1
        fixed=torch.randint(2,(tasks,cfg['flipping_bits']),dtype=torch.float32).repeat_interleave(cfg['flip_after'],0)[:count]
        random=torch.randint(2,(count,cfg['inputs']-cfg['flipping_bits']),dtype=torch.float32)
        x=torch.cat((fixed,random),1);parts=[]
        with torch.no_grad():
            for batch in x.split(10000):parts.append(target.predict(batch)[0])
        y=torch.cat(parts)
    torch.save(dict(identity=identity,x=x,y=y),path)
    return x,y
