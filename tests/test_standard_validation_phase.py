import json
from pathlib import Path
import pytest
import torch


def test_h3_each_task_has_early_and_regular_opportunities():
    from experiments.standard_imagenet import intervention_due
    cfg=json.loads(Path('configs/standard_cl/h3_imagenet_small.json').read_text())
    last=0;opportunities=[]
    for task in range(100):
        for local in range(1,601):
            step=task*600+local
            if intervention_due(step,last,task*600,min(local*100,128),1200,cfg):
                opportunities.append((task,local));last=step
    assert len(opportunities)==300
    assert all([step for t,step in opportunities if t==task]==[2,202,402] for task in range(100))
    old=json.loads(Path('configs/standard_cl/h3_imagenet_cadence4000.json').read_text())
    assert old['intervention_every']==4000 and old['task_start_intervention'] is False



def test_geometry_pilot_budget_records_all_failed_windows(monkeypatch):
    pytest.importorskip('avalanche')
    from types import SimpleNamespace
    from experiments.standard_online import GeometryPlugin,GeometrySelectionFailed
    cfg=json.loads(Path('configs/standard_cl/h2_online_cifar100.json').read_text());cfg['intervention_every']=1
    plugin=GeometryPlugin(cfg,0);plugin.task=1
    monkeypatch.setattr(plugin,'select_geometry',lambda *a:dict(selected_weight=None,status='NO_CERTIFIED_NONZERO_CANDIDATE',seconds=.5,trials=[]))
    x=torch.ones(2,2);y=torch.tensor([0,1]);mask=torch.tensor([False,True])
    strategy=SimpleNamespace(mb_x=x,mb_y=y,mbatch=[x,y,mask,None],clock=SimpleNamespace(train_iterations=0))
    for index in range(3):
        strategy.clock.train_iterations=index
        if index==2:
            with pytest.raises(GeometrySelectionFailed):plugin.after_update(strategy)
        else:plugin.after_update(strategy)
    assert plugin.selection is None and len(plugin.selection_attempts)==3
    assert len(plugin.engine.events)==plugin.engine.applications==3
    assert sum(r['seconds'] for r in plugin.engine.events)==1.5
