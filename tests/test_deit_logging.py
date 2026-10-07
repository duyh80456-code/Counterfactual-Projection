"""Console labels/provenance match Res18 style without mutating training state."""
import json
import torch
from experiments.deit_logging import emit_event, emit_epoch


def test_tagged_epoch_is_saved_and_does_not_change_history_or_rng(tmp_path, capsys, monkeypatch):
    import experiments.deit_logging as logger
    monkeypatch.setattr(logger.time, 'perf_counter', lambda: 12.)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *_a: (_ for _ in ()).throw(AssertionError('CPU used CUDA')))
    row = {'epoch': 302, 'validation_accuracy': .731333, 'report_best_improved': False,
           'controller_anchor_improved': True, 'controller_anchor_reason': 'same_accuracy_lower_loss',
           'controller_stall_counter': 0}
    before = torch.get_rng_state().clone()
    emit_epoch('e_driven_o_raw', row, torch.device('cpu'), 10., tmp_path)
    console = capsys.readouterr().out.strip()
    assert json.loads(console)['e_driven_o_raw']['epoch_seconds'] == 2.
    assert json.loads(console)['e_driven_o_raw']['peak_gpu_memory'] == 0
    assert (tmp_path / 'console.jsonl').read_text().strip() == console
    assert 'epoch_seconds' not in row and torch.equal(before, torch.get_rng_state())
    emit_event('deit_rollback', {'anchor_epoch': 302}, tmp_path)
    assert len((tmp_path / 'console.jsonl').read_text().splitlines()) == 2
