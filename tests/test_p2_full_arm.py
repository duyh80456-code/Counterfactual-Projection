"""The real main iCaRL-full entry point must dispatch to the opt-in swap rule."""
import json

from experiments import main_cil as main
from methods import vna_certswap


def test_main_icarl_full_p2_routes_certswap(tmp_path,monkeypatch):
    original_after=main.Intervention.after_step
    original_probe=main.fixed_probe
    original_sites=main.sites_for
    def window(self,hook,learner,optimizer):
        current=hook.current_batch
        hook.current_batch=tuple(value[:2] for value in current)
        try:return original_after(self,hook,learner,optimizer)
        finally:hook.current_batch=current
    monkeypatch.setattr(main.Intervention,'after_step',window)
    monkeypatch.setattr(main,'fixed_probe',lambda learner,size:original_probe(learner,min(size,2)))
    monkeypatch.setattr(main,'sites_for',lambda model:original_sites(model)[:1])
    calls=[]
    def leaf(engine,model,sites,proposals,batch,spec,ctx,optimizer,iteration,values):
        assert engine.arm=='vna_gir' and engine.gir_rule=='certswap' and engine.task==2
        assert engine.policy_version=='certswap_t32_v1' and len(batch[0])==2
        assert sites and model.native.convnet.stage_1 and optimizer.state
        calls.append(dict(iteration=iteration,classes=ctx['known_classes'],sites=[s.module_name for s in sites]))
        return dict(sites=[],accepted=[],trials=[],rejections={},seconds=0.)
    monkeypatch.setattr(vna_certswap,'run',leaf)
    args=main.parser().parse_args(['--arm','icarl_full','--fake-data','--device','cpu',
        '--workers','0','--solver','gram','--curvature','power10','--increment','10',
        '--warmup-epochs','1','--gir-rule','certswap','--no-certswap-log-schur','--out',str(tmp_path)])
    summary=main.run(args)
    assert summary['status']=='COMPLETED' and calls
    assert all(c['classes']==10 for c in calls)
    identity=summary['identity']
    assert identity['arm']=='icarl_full' and identity['gir_rule']=='certswap'
    assert identity['policy_version']=='certswap_t32_v1' and identity['training_protocol']=='P2'
    events=json.loads((tmp_path/'interventions.json').read_text())
    assert any('certswap' in event for event in events)
