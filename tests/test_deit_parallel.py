"""Two GPU slots execute independent processes without changing fork/protocol."""
import json
import sys
import threading
from pathlib import Path

import pytest
from experiments.run_deit_all_arms import (run_jobs_parallel, gpu_slots, child_environment, command_on_device)


def commands(names):
    return {name: ['python', '--method', name, '--device', 'cuda:0', '--plateau-checkpoint', '/theta219.pt',
                   '--plateau-checkpoint-hash', 'same-fork-hash', '--resume', f'/arms/{name}/latest.pt'] for name in names}


def test_two_slots_overlap_refill_and_opt_e_barrier(tmp_path):
    names = ['vanilla_continue', 'o_projection_only', 'e_driven_o_raw', 'opt_e', 'e_driven_o_normalized']
    second_started, third_started = threading.Event(), threading.Event()
    lock = threading.Lock()
    active, maximum, starts, finished = {}, [0], [], set()
    def runner(command, check, env):
        method = command[command.index('--method') + 1]
        slot = env['CUDA_VISIBLE_DEVICES']
        assert command[command.index('--plateau-checkpoint') + 1] == '/theta219.pt'
        assert command[command.index('--plateau-checkpoint-hash') + 1] == 'same-fork-hash'
        assert '--resume' in command
        assert command[command.index('--device') + 1] == ('cpu' if method == 'vanilla_continue' else 'cuda:0')
        with lock:
            starts.append(method)
            if method not in ('vanilla_continue', 'opt_e'):
                assert 'vanilla_continue' in finished
            if method == 'opt_e':
                assert set(names) - {'opt_e'} == finished
            if slot:
                active[slot] = active.get(slot, 0) + 1
                assert active[slot] == 1
                maximum[0] = max(maximum[0], sum(active.values()))
        try:
            if method == 'o_projection_only':
                assert second_started.wait(3), 'second GPU did not start concurrently'
            elif method == 'e_driven_o_raw':
                second_started.set()
                assert third_started.wait(3), 'free GPU did not refill while the other was busy'
            elif method == 'e_driven_o_normalized':
                third_started.set()
        finally:
            with lock:
                if slot:
                    active[slot] -= 1
                finished.add(method)
    statuses = run_jobs_parallel(commands(names), tmp_path / 'status.json', [0, 1], runner=runner, parent_env={})
    assert all(row['status'] == 'completed' for row in statuses.values())
    assert maximum[0] == 2 and starts[0] == 'vanilla_continue' and starts[-1] == 'opt_e'
    assert statuses['o_projection_only']['gpu_slot'] == 0
    assert statuses['e_driven_o_raw']['gpu_slot'] == 1
    assert statuses['e_driven_o_normalized']['gpu_slot'] == 0
    assert statuses['vanilla_continue']['gpu_slot'] is None
    assert json.loads((tmp_path / 'status.json').read_text()) == statuses


def test_failure_isolated_and_single_gpu_supported(tmp_path):
    calls = []
    def runner(command, check, env):
        method = command[command.index('--method') + 1]
        calls.append(method)
        assert env['CUDA_VISIBLE_DEVICES'] == '7'
        if method == 'e_driven_o_raw':
            raise RuntimeError('intentional arm failure')
    names = ['opt_e', 'e_driven_o_raw', 'persistent_growth']
    result = run_jobs_parallel(commands(names), tmp_path / 'status.json', [1], runner=runner,
                               parent_env={'CUDA_VISIBLE_DEVICES': '3,7'})
    assert calls == ['e_driven_o_raw', 'persistent_growth', 'opt_e']
    assert result['e_driven_o_raw']['status'] == 'failed'
    assert result['persistent_growth']['status'] == result['opt_e']['status'] == 'completed'


def test_gpu_discovery_and_visible_device_mapping(monkeypatch):
    import experiments.run_deit_all_arms as suite
    monkeypatch.setattr(suite.torch.cuda, 'device_count', lambda: 2)
    assert gpu_slots('auto', 'cuda:0') == [0, 1]
    assert gpu_slots('1', 'cuda:0') == [1]
    assert gpu_slots('auto', 'cpu') == []
    for invalid in ('0,0', '2', '-1'):
        with pytest.raises(ValueError): gpu_slots(invalid, 'cuda:0')
    assert child_environment(1, {'CUDA_VISIBLE_DEVICES': 'GPU-first,GPU-second'})['CUDA_VISIBLE_DEVICES'] == 'GPU-second'
    assert child_environment(None, {})['CUDA_VISIBLE_DEVICES'] == ''
    assert command_on_device(['python', '--device', 'cuda:1'], 1) == ['python', '--device', 'cuda:0']
    monkeypatch.setattr(suite.torch.cuda, 'device_count', lambda: 1)
    assert gpu_slots('auto', 'cuda:0') == [0]


def test_real_child_process_console_and_failure(tmp_path, capsys):
    jobs = {'a': [sys.executable, '-c', "print('hello from arm')"],
            'b': [sys.executable, '-c', 'import sys; sys.exit(2)']}
    result = run_jobs_parallel(jobs, tmp_path / 'status.json', [0, 1], parent_env={})
    assert result['a']['status'] == 'completed' and result['b']['status'] == 'failed'
    assert '[GPU0 a] hello from arm' in capsys.readouterr().out
