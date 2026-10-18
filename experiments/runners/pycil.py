"""Pinned PyCIL location, upstream imports and the original iCaRL recipe."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT/'third_party/PyCIL'
UPSTREAM_PIN = 'f3509b8ca3f20660ce4aa13f19d5283de81b4b35'


def upstream_imports():
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    if not (UPSTREAM/'models/icarl.py').exists(): raise FileNotFoundError('pinned PyCIL absent; no automatic fetch')
    pin = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if pin != UPSTREAM_PIN: raise ValueError('PyCIL commit does not match registered pin')
    if str(UPSTREAM) not in sys.path: sys.path.insert(0, str(UPSTREAM))

def recipe():
    args = json.loads((UPSTREAM/'exps/icarl.json').read_text())
    args['device'] = ['0']
    return args

