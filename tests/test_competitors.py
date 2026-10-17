import copy
import json
import numpy as np
import pytest
import torch
from experiments import competitors as c


@pytest.mark.parametrize('method', c.CIL_METHODS)
def test_upstream_method_one_epoch(tmp_path, method, monkeypatch):
    params, source, overrides = c.cil_recipe(method, 'b50inc10')
    original = json.loads((c.cil.UPSTREAM/source).read_text())
    assert set(overrides) == {k for k in params if params[k] != original.get(k)}
    assert params['fixed_memory'] == original['fixed_memory']
    assert params['memory_size'] == 2000
    assert params['init_cls'] == (100 if method == 'joint' else 50)
    args = c.parser().parse_args(['cil','--method',method,'--device','cpu',
        '--fake-data','--out',str(tmp_path/method)])
    result = c.run_cil(args)
    protocol = json.loads((tmp_path/method/'protocol.json').read_text())
    _, _, expected = c.cil_recipe(method, 'inc10', device='cpu')
    assert protocol['config_overrides'] == expected
    assert result['status'] == 'COMPLETED'
    assert result['tasks_run'] == 1
    assert result['parameter_count'] > 0
    assert result['reproduction']['status'] == 'INVALID'
    if method == 'finetune':
        expected_state = torch.load(tmp_path/method/'latest.pt', weights_only=False)['model']._network.state_dict()
        args.out = str(tmp_path/'interrupted')
        calls = 0
        def interrupt(checkpoint):
            nonlocal calls
            calls += 1
            if calls == 2:
                checkpoint()
                raise SystemExit(75)
        with monkeypatch.context() as patch:
            patch.setattr(c, 'pause_if_due', interrupt)
            with pytest.raises(SystemExit) as paused: c.run_cil(args)
            assert paused.value.code == 75
        args.resume = True
        resumed = c.run_cil(args)
        actual_state = torch.load(tmp_path/'interrupted'/'latest.pt', weights_only=False)['model']._network.state_dict()
        assert resumed['tasks'] == result['tasks']
        assert all(torch.equal(expected_state[k], actual_state[k]) for k in expected_state)


def test_redo_resets_upstream_dormant_neuron():
    c.lop.upstream_imports()
    from lop.nets.ffnn import FFNN
    from lop.algos.bp import Backprop
    intervention = c.ReDoHook()
    with intervention.install(): learner = Backprop(FFNN(2, num_features=2), step_size=.01)
    with torch.no_grad():
        learner.net.layers[0].weight[0].fill_(-1)
        learner.net.layers[0].bias[0].zero_()
        learner.net.layers[0].weight[1].fill_(1)
    learner.gnt.steps_since_last_redo = 999
    learner.previous_features = [torch.tensor([[0.,2.]])]
    intervention.after(None, learner)
    assert learner.gnt.steps_since_last_redo == 0
    assert torch.count_nonzero(learner.net.layers[2].weight[:,0]) == 0
    assert not torch.equal(learner.net.layers[0].weight[0], torch.tensor([-1.,-1.]))


@pytest.mark.parametrize('arm', ['l2','snp','redo'])
def test_b5_competitor_sweep_and_resume(tmp_path, arm):
    args = c.parser().parse_args(['b5','--arm',arm,'--fake-data','--out',str(tmp_path)])
    result = c.run_b5(args)
    assert [r['seed'] for r in result['variants']] == [0,1,2]
    assert all(r['steps'] == 1000 for r in result['variants'])
    args.resume = True
    assert c.run_b5(args) == result
