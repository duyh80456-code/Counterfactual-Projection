"""Res18-style tagged JSON events for console and resumable JSONL logs."""
from __future__ import annotations

import json
from pathlib import Path
import time
import torch


def emit_event(event, payload, output=None):
    line = json.dumps({event: payload}, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        path = Path(output) / 'console.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(line + '\n')


def emit_epoch(event, row, device, started, output=None):
    # Wall time/memory are observations, kept out of deterministic training history.
    observed = {**row, 'epoch_seconds': time.perf_counter() - started,
                'peak_gpu_memory': int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0}
    emit_event(event, observed, output)
