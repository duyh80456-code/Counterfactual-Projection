"""Fetch the two RepDistiller teachers, verify bytes, then gate with its validate."""
import argparse
import hashlib
import json
from contextlib import nullcontext
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import urllib.request
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import torch
from adapters.repdistiller_hooks import upstream, UPSTREAM
from experiments.stage_b5 import write_json

TEACHERS = {
    'resnet32x4': (59554438, '22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f', 79.),
    'wrn_40_2': (18113193, '41100a8e8478b326645eb9aab09c18fc8ed34c1ab4004d658cb652125a81ca18', 75.),
}


def verify(path, size, sha256):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): digest.update(chunk)
    if path.stat().st_size != size or digest.hexdigest() != sha256:
        raise ValueError(f'teacher size/SHA256 mismatch: {path}')
    return digest.hexdigest()


def fetch(name):
    size, digest, _ = TEACHERS[name]
    path = UPSTREAM/'save/models'/f'{name}_vanilla/ckpt_epoch_240.pth'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.download')
        try:
            urllib.request.urlretrieve(f'http://shape2prog.csail.mit.edu/repo/{name}_vanilla/ckpt_epoch_240.pth', temporary)
            verify(temporary, size, digest)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    verify(path, size, digest)
    return path


def teacher_gate(name, path, data_dir, device, *, max_batches=None, workers=4):
    size, digest, threshold = TEACHERS[name]
    verify(path, size, digest)
    with upstream() as models:
        from dataset import cifar100
        from helper.loops import validate
        original = cifar100.get_data_folder
        try:
            cifar100.get_data_folder = lambda: str(Path(data_dir).resolve())
            _, loader = cifar100.get_cifar100_dataloaders(batch_size=64, num_workers=workers)
            if max_batches:
                from experiments.cifar_pilot import LimitedLoader
                loader = LimitedLoader(loader, max_batches)
            model = models[name](num_classes=100)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
            model.to(device).eval()
            cuda = patch('torch.cuda.is_available', return_value=False) if device == 'cpu' else nullcontext()
            if device.startswith('cuda'): torch.cuda.set_device(torch.device(device))
            with cuda:
                acc, _, _ = validate(loader, model, torch.nn.CrossEntropyLoss().to(device), SimpleNamespace(print_freq=100))
        finally:
            cifar100.get_data_folder = original
    return {'teacher': name, 'sha256': digest, 'size': size, 'top1': float(acc),
            'threshold': threshold, 'test_batches': len(loader), 'report_only': bool(max_batches),
            'status': 'PILOT' if max_batches else 'PASS' if float(acc) >= threshold else 'FAIL',
            'threshold_met': float(acc) >= threshold}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', choices=TEACHERS, required=True)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--gate-out', default='outputs/stageB/teacher_gate.json')
    args = p.parse_args()
    path = fetch(args.teacher)
    row = teacher_gate(args.teacher, path, args.data_dir, args.device)
    out = Path(args.gate_out)
    rows = json.loads(out.read_text()) if out.exists() else {}
    rows[args.teacher] = row
    write_json(out, rows)
    if row['status'] != 'PASS': raise ValueError(f'teacher gate failed: {row}')


if __name__ == '__main__': main()
