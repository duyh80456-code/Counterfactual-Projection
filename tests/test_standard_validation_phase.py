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
    from experiments.standard_online import GeometryPlugin
    cfg=json.loads(Path('configs/standard_cl/h2_online_cifar100.json').read_text());cfg['intervention_every']=1
    plugin=GeometryPlugin(cfg,0);plugin.task=1
    monkeypatch.setattr(plugin,'select_geometry',lambda *a:dict(selected_weight=None,status='NO_CERTIFIED_NONZERO_CANDIDATE',seconds=.5,trials=[]))
    x=torch.ones(2,2);y=torch.tensor([0,1]);mask=torch.tensor([False,True])
    strategy=SimpleNamespace(mb_x=x,mb_y=y,mbatch=[x,y,mask,None],clock=SimpleNamespace(train_iterations=0))
    for index in range(3):
        strategy.clock.train_iterations=index
        plugin.after_update(strategy)
    assert plugin.selection['status']=='FALLBACK_ER' and len(plugin.selection_attempts)==3
    strategy.clock.train_iterations=100
    plugin.after_update(strategy)
    assert len(plugin.engine.events)==plugin.engine.applications==3
    assert sum(r['seconds'] for r in plugin.engine.events)==1.5


def test_geometry_evidence_distinguishes_zero_and_fallback():
    from scripts.collect_standard_cl import geometry_evidence
    assert geometry_evidence(dict(status='SELECTED',selected_weight=0.),1.,True)=='CALIBRATED_ZERO_GEOMETRY'
    assert geometry_evidence(dict(status='SELECTED',selected_weight=10.),1.,True)=='CALIBRATED_POSITIVE_GEOMETRY'
    assert geometry_evidence(dict(status='FALLBACK_ER',selected_weight=None),1.,True)=='VNA_PREFIX_THEN_ER_FALLBACK'


def test_ablation_weight_lock_checks_authority_and_dataset(tmp_path):
    import hashlib
    from copy import deepcopy
    from scripts.build_kaggle_standard_compare import lock_geometry_config
    cfg=json.loads(Path('configs/standard_cl/h2_online_cifar100.json').read_text())
    raw=tmp_path/'cifar-100-python';raw.mkdir();fingerprint={}
    for name in ('train','test','meta'):
        (raw/name).write_bytes(name.encode());fingerprint[name]=hashlib.sha256(name.encode()).hexdigest()
    report=dict(status='SELECTED',selected_weight=.1,seed=0,source_commit='a'*40,
        benchmark_config=cfg,dataset_fingerprint=fingerprint,validation_sha256='b'*64)
    authority=tmp_path/'geometry_selection.json';authority.write_text(json.dumps(report))
    solver=deepcopy(cfg);solver['correction']['preconditioner_probes']=0
    reference=lock_geometry_config(cfg,authority,tmp_path,'a'*40)
    locked=lock_geometry_config(solver,authority,tmp_path,'a'*40)
    assert reference['geometry_weight']==locked['geometry_weight']==.1
    assert reference['geometry_authority']==locked['geometry_authority'] and 'geometry_selection' not in locked
    with pytest.raises(ValueError,match='source mismatch'):lock_geometry_config(solver,authority,tmp_path,'c'*40)
    (raw/'train').write_bytes(b'changed')
    with pytest.raises(ValueError,match='dataset mismatch'):lock_geometry_config(solver,authority,tmp_path,'a'*40)


def test_collector_never_averages_zero_and_fallback_with_geometry(tmp_path):
    from scripts.collect_standard_cl import collect
    selections=[dict(status='SELECTED',selected_weight=0.),dict(status='SELECTED',selected_weight=1.),dict(status='FALLBACK_ER',selected_weight=None)]
    for seed,selection in enumerate(selections):
        out=tmp_path/str(seed);out.mkdir()
        protocol=dict(branch='H2',arm='h2',seed=seed,tasks=1,config=dict(geometry_weight=1.,geometry_selection={'weights':[0.,1.]}),dataset_fingerprint={'train':'same'})
        summary=dict(protocol=protocol,status='COMPLETED',tasks=[{'total':70.+seed}],geometry_selection=selection,
            average_incremental_accuracy=70.+seed,final_accuracy=70.+seed,forgetting=0.,backward_transfer=0.,gpu_hours=0.)
        (out/'protocol.json').write_text(json.dumps(protocol));(out/'summary.json').write_text(json.dumps(summary))
    rows=collect([tmp_path])
    assert len(rows)==3 and all(len(r['seeds'])==1 for r in rows)
    assert {r['geometry_evidence'] for r in rows}=={'CALIBRATED_ZERO_GEOMETRY','CALIBRATED_POSITIVE_GEOMETRY','VNA_PREFIX_THEN_ER_FALLBACK'}


def test_numerical_ablation_cannot_recalibrate_without_authority():
    pytest.importorskip('avalanche')
    from experiments.standard_online import GeometryPlugin
    cfg=json.loads(Path('configs/standard_cl/h2_online_cifar100_solver_ablation.json').read_text())
    assert 'geometry_selection' not in cfg
    with pytest.raises(ValueError,match='frozen geometry authority'):GeometryPlugin(cfg,0)
