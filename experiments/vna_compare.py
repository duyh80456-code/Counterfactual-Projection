"""Compare finished Stage 2a arms across notebook directories and archives."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tarfile
import zipfile

from experiments.shared_protocol import atomic_json_save

ARMS = ('sgd', 'vna_tiny_V', 'ngd', 'vna_random', 'vna_tiny_U', 'sgd_cm')
VNA_ARMS = set(ARMS) - {'sgd', 'sgd_cm'}


def hash_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def save_run_config(path, config):
    digest = hash_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != dict(config, config_hash=digest):
        raise ValueError('Existing run_config.json differs; use a fresh output directory')
    atomic_json_save(dict(config, config_hash=digest), path)
    return digest


def load_run_config(path):
    config = json.loads(Path(path).read_text())
    digest = config.pop('config_hash')
    if hash_config(config) != digest:
        raise ValueError('run_config config_hash mismatch')
    return config, digest


def recipe_protocol(protocol):
    # Arm/selection are intentional treatment differences; sgd_cm's schedule is
    # derived from the common epochs_scale. All other parser fields must agree.
    scale = protocol['epochs_scale'] if protocol['arm'] == 'sgd_cm' else 1.
    expected = {'total_epochs': round(protocol['epochs'] * scale),
                'milestones': [round(100 * scale), round(150 * scale)]}
    diff = _diff(expected, {k: protocol.get(k) for k in expected})
    if diff: raise ValueError('Recipe schedule mismatch; diff=' + json.dumps(diff, sort_keys=True))
    return {k: v for k, v in protocol.items()
            if k not in ('arm', 'selection', 'total_epochs', 'milestones', 'solver')}


def _diff(a, b):
    return {k: {'first': a.get(k), 'other': b.get(k)}
            for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}


def _documents(root):
    """Read JSON only, without extracting archive paths or large checkpoints."""
    for directory, _, files in os.walk(root, followlinks=True):
        for filename in sorted(files):
            path = Path(directory) / filename
            if filename == 'final.json':
                yield str(path), json.loads(path.read_text()), lambda name, p=path: json.loads((p.parent / name).read_text())
            elif filename.lower().endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
                if zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path) as archive:
                        names = set(archive.namelist())
                        for name in sorted(names):
                            if Path(name).name == 'final.json':
                                base = name.rsplit('/', 1)[0] if '/' in name else ''
                                read = lambda n, a=archive, b=base: json.loads(a.read(b + '/' + n if b else n))
                                yield f'{path}:{name}', json.loads(archive.read(name)), read
                else:
                    with tarfile.open(path, 'r:*') as archive:
                        members = {m.name: m for m in archive.getmembers() if m.isfile()}
                        for name in sorted(members):
                            if Path(name).name == 'final.json':
                                base = name.rsplit('/', 1)[0] if '/' in name else ''
                                def read(n, a=archive, b=base):
                                    key = b + '/' + n if b else n
                                    return json.load(a.extractfile(members[key]))
                                yield f'{path}:{name}', json.load(archive.extractfile(members[name])), read


def collect_results(root, seed, input_root='/kaggle/input'):
    records = []
    for location in dict.fromkeys([str(root), str(input_root)]):
        for source, record, read in _documents(location):
            if record.get('seed') != seed:
                continue
            protocol = record.get('protocol')
            if not isinstance(protocol, dict):
                raise ValueError(f'Missing recipe protocol: {source}')
            arm = record['arm']
            if arm == 'vna':
                arm = 'vna_' + record['selection']
            if arm not in ARMS:
                continue
            history = read('history.json')
            if record.get('epoch') != protocol['total_epochs'] or not history or history[-1]['epoch'] != protocol['total_epochs']:
                continue
            records.append(dict(arm=arm, record=record, history=history, source=source))
    # Check every candidate before deduplication, so stale conflicting archives
    # cannot silently overwrite an incompatible result.
    if records:
        first = records[0]
        reference = recipe_protocol(first['record']['protocol'])
        vna = None
        for item in records:
            diff = _diff(reference, recipe_protocol(item['record']['protocol']))
            if diff:
                raise ValueError(f'Recipe protocol mismatch: {first["source"]} vs {item["source"]}; diff={json.dumps(diff, sort_keys=True)}')
            if item['arm'] in VNA_ARMS:
                r = item['record']
                config, digest = r.get('run_config'), r.get('config_hash')
                if not isinstance(config, dict) or not digest or hash_config(config) != digest:
                    raise ValueError(f'Invalid config_hash: {item["source"]}')
                if vna and digest != vna['record']['config_hash']:
                    diff = _diff(vna['record']['run_config'], config)
                    raise ValueError(f'config_hash mismatch: {vna["source"]} vs {item["source"]}; diff={json.dumps(diff, sort_keys=True)}')
                vna = item
    selected = {}
    for item in records:
        old = selected.get(item['arm'])
        if old and old['record'] != item['record']:
            raise ValueError(f'Conflicting finished results for {item["arm"]}: {old["source"]} vs {item["source"]}; diff={json.dumps(_diff(old["record"], item["record"]), sort_keys=True)}')
        selected[item['arm']] = item
    return selected



def _curves(records):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for arm, item in records.items():
        history = item['history']; epochs = [r['epoch'] for r in history]
        for ax, values in zip(axes, [[r['train']['loss'] for r in history],
                                     [r['eval']['accuracy'] for r in history],
                                     [r['mean_rho'] for r in history]]):
            ax.plot(epochs, values, label=arm)
    for ax, title in zip(axes, ['Train loss', 'Eval accuracy', 'rho']):
        ax.set(title=title, xlabel='epoch'); ax.grid(alpha=.3)
    if records: axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def compare(root, seed, *, input_root='/kaggle/input', output=None, plot=True):
    import pandas as pd
    records = collect_results(root, seed, input_root)
    baseline = records.get('sgd', {}).get('record')
    rows = []
    for arm in ARMS:
        if arm not in records:
            continue
        r = records[arm]['record']
        rows.append(dict(arm=arm, test_acc=r['test']['accuracy'], test_loss=r['test']['loss'],
                         delta_vs_sgd_pp=100*(r['test']['accuracy']-baseline['test']['accuracy']) if baseline else None,
                         cost_vs_sgd=r['gpu_hours']/baseline['gpu_hours'] if baseline and baseline['gpu_hours'] else None,
                         params=r['params'], gpu_hours=r['gpu_hours'], wall_hours=r['wall_hours'],
                         code_commits=sorted({h['commit'] for h in r['code_history']})))
    frame = pd.DataFrame(rows)
    missing = [a for a in ARMS if a not in records]
    fig = _curves(records) if plot else None
    if output is not None:
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output/'results.csv', index=False)
        if fig is not None: fig.savefig(output/'curves.png')
        atomic_json_save({'seed': seed, 'available': list(records), 'missing': missing}, output/'comparison.json')
    return frame, missing, fig


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--input-root', default='/kaggle/input'); p.add_argument('--output', required=True)
    args = p.parse_args()
    frame, missing, _ = compare(args.root, args.seed, input_root=args.input_root, output=args.output)
    print(frame.to_string(index=False)); print('Missing arms:', missing)
