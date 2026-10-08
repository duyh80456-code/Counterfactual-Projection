"""Build one independent, matched rollback comparison notebook per DeiT seed."""
import copy
import json
from pathlib import Path

ARMS = ('vanilla_rollback', 'o_projection_only', 'o_projection_only_rollback', 'e_driven_o_raw')
TEMPLATE = Path('notebooks/kaggle_deit_tiny_seed1_all_arms.ipynb')


def build(seed):
    notebook = copy.deepcopy(json.loads(TEMPLATE.read_text()))
    notebook['cells'][0]['source'] = f'''# DeiT-Tiny CIFAR-100 — seed {seed}: projection and rollback comparison

Run this notebook independently from the other seed notebook. Attach CIFAR-100,
enable Internet and T4x2, and enable Kaggle Secret `github_token`.
Optional resume input: the output archive from this same seed/config.
Seed1 checkpoints cannot initialize seed{seed}'s experiment.

## Phase 1: Vanilla plateau
Train Vanilla from scratch for this seed; strict validation accuracy best resets
the stall. After150 epochs without exceeding the historical best, confirm plateau.
No minimum epoch and no requirement to finish the cosine schedule. Fork all four
arms from exactly that best model, AdamW and scheduler, RNG and loader state.
The150 Vanilla confirmation epochs are retained as an observed baseline.

## Phase 2: four arms, 150 training epochs each

| Arm | Projection | Rollback |
| --- | --- | --- |
| `vanilla_rollback` | None; genuine Vanilla training | Yes |
| `o_projection_only` | O-only, fixed `blocks.11.mlp` | No |
| `o_projection_only_rollback` | Same O-only as above | Yes |
| `e_driven_o_raw` | Raw E-WHERE followed by projection into O | Yes |

Vanilla with rollback is a new150-epoch continuation from the fork; the ordinary
Vanilla confirmation trajectory cannot substitute for it. This tests the
controller effect separately, and O-only with/without rollback isolates that effect.

Shared rollback rule: accuracy increases OR equal accuracy/lower loss updates
anchor; otherwise increment stall once. At10 stalled epochs restore model,
optimizer and scheduler to the latest anchor, preserving current RNG/data stream.
No new intervention on rollback. Report best uses strict accuracy only and keeps
loss from the same best epoch. Scientific escape compares post-fork best to this
seed's historical Vanilla best, never to the other seed or another arm.

Recipe: batch64, AdamW LR5e-4/WD.05, warmup5 + fixed cosine300 to LR5e-6;
bias/norm/CLS/position have no weight decay. CP: rank4, projection32,
scales(.0125,.025,.05), stats256, WHERE3x32, gate32, CG200. No random-control,
normalized-WHERE, persistent-growth or Opt-E arm runs in these notebooks.
The validation evaluation split is3000 samples;2000 reserved samples are unused
for trigger/selection. Official test is unused.

Phase1 uses GPU0; Phase2 queues four isolated arms across two GPUs, one arm per GPU.
The next queued arm fills a free GPU; no Opt-E barrier is relevant here.
Each epoch saves full state and prints method/GPU, train/val, LR, report best,
anchor, stall counters and rollback. On resume completed arms are skipped and
partial arms continue without repeating their projection. Output is archived in
`finally`; abrupt kernel termination may leave only the latest completed checkpoints.
'''.splitlines(keepends=True)
    config = ''.join(notebook['cells'][2]['source'])
    config = config.replace('SEED = 1', f'SEED = {seed}')
    config = config.replace('_all_arms_v5', '_rollback_comparison_v5')
    config = config.replace('SEED = '+str(seed), 'SEED = '+str(seed)+'\nARMS = '+repr(ARMS), 1)
    notebook['cells'][2]['source'] = config.splitlines(keepends=True)
    discovery = ''.join(notebook['cells'][3]['source'])
    discovery = discovery.replace('from experiments.run_deit_all_arms import ALL_ARMS\n', '')
    discovery = discovery.replace('arms=ALL_ARMS', 'arms=ARMS')
    notebook['cells'][3]['source'] = discovery.splitlines(keepends=True)
    command = ''.join(notebook['cells'][4]['source'])
    command = command.replace("'--gpu-devices', GPU_DEVICES]", "'--gpu-devices', GPU_DEVICES, '--arms', ','.join(ARMS)]")
    notebook['cells'][4]['source'] = command.splitlines(keepends=True)
    notebook['cells'][5]['source'] = '''# Compare post-fork strict bests; never replace an arm by another arm's best.
from experiments.shared_protocol import atomic_json_save
comparison = {}
for method in ARMS:
    path = OUTPUT / 'arms' / method / 'result.json'
    if path.exists():
        result = json.loads(path.read_text())
        comparison[method] = {key: result[key] for key in (
            'fork_epoch', 'historical_best_accuracy', 'report_best_accuracy',
            'report_best_loss', 'report_best_epoch', 'delta_vs_historical_best',
            'scientific_escape', 'intervention_count')}
        comparison[method]['rollback_count'] = len(result['rollback_events'])
if 'vanilla_rollback' in comparison:
    control = comparison['vanilla_rollback']['report_best_accuracy']
    for row in comparison.values():
        row['delta_vs_vanilla_rollback'] = row['report_best_accuracy'] - control
atomic_json_save(comparison, OUTPUT / 'rollback_comparison.json')
print(json.dumps(comparison, indent=2))
status_path = OUTPUT / 'arm_status.json'
if status_path.exists():
    print('Status:', status_path.read_text())
'''.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    notebook['metadata'].setdefault('kaggle', {}).update(isGpuEnabled=True, isInternetEnabled=True)
    target = Path(f'notebooks/kaggle_deit_tiny_seed{seed}_rollback_comparison.ipynb')
    target.write_text(json.dumps(notebook, indent=1) + '\n')
    return target


if __name__ == '__main__':
    for seed in (0, 2):
        print(build(seed))
