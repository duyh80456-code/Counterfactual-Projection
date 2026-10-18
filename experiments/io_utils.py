"""Atomic artifacts and native working directories."""
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import pickle
import torch

@contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(path)
    try: yield
    finally: os.chdir(old)

def atomic_pickle(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f: pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, default=str)+'\n'); tmp.replace(path)

def write_csv(path, rows):
    rows = list(rows)
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = Path(path).with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)

def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)
