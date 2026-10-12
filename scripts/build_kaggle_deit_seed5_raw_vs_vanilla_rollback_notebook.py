"""Seed5: exactly E→O raw and genuine Vanilla rollback from a common fork."""
import copy
import json
from pathlib import Path


def build():
    notebook=copy.deepcopy(json.loads(Path('notebooks/kaggle_deit_tiny_seed2_rollback_comparison.ipynb').read_text()))
    notebook['cells'][0]['source']='''# DeiT-Tiny seed5 — E→O raw top1 vs Vanilla + rollback

Attach CIFAR-100; enable Internet, T4x2 and Kaggle Secret `github_token`.
Optional resume: full output archive from this same seed/recipe. Compatible
seed5 E→O progress from the earlier raw-vs-Bypass notebook can be reused only
with its original hash-bound fork/raw reference and matching configuration.
Each requested arm resumes independently. No Bypass, O-only, A3/A4 or extra arm.

Phase1 trains Vanilla for seed5 until150 epochs without a strict validation
accuracy best. No minimum epoch and no requirement to finish schedule300.
Reload that historical-best model, AdamW, scheduler and RNG/loader state.

Phase2 runs two independent150-epoch trajectories from identical full fork:
- GPU0: `e_driven_o_raw`, original TINY top1 WHERE over12 MLPs, one E→O projection.
- GPU1: `vanilla_rollback`, genuine Vanilla training without E/projection/moment reset.

Both use patience10. Higher accuracy OR an exact accuracy tie with lower loss
updates the controller anchor and resets stall; otherwise stall increments once.
Rollback restores model/optimizer/scheduler and retains the current RNG/loader
stream. No retrigger. Report best uses strict accuracy and paired loss.
Scientific escape compares observed post-fork best to this seed's historical best.
Vanilla rollback is newly trained, not the observed Vanilla plateau-confirmation
window. On resume completed arms skip training and remaining arms use free GPU slots.

Recipe: batch64, AdamW LR5e-4/WD.05, warmup5 + cosine300, plateau150, horizon150,
max_epoch800. E→O rank4, stats256, WHERE3x32, projection32, gate32, CG200,
scales(.0125,.025,.05). Evaluation uses3000 validation samples;2000 reserved
samples do not trigger/select anything. Official test is unused.

Every epoch saves full state including anchor/counter, optimizer, scheduler,
RNG and loader. Download the full output tarball for another session.
The comparison uses completed current arms with identical fork/protocol/horizon;
failed or unrequested results do not enter the accuracy delta.
'''.splitlines(keepends=True)
    config=''.join(notebook['cells'][2]['source']).replace('SEED = 2','SEED = 5')
    config=config.replace("ARMS = ('vanilla_rollback', 'o_projection_only', 'o_projection_only_rollback', 'e_driven_o_raw')",
        "ARMS = ('e_driven_o_raw', 'vanilla_rollback')")
    config=config.replace('_rollback_comparison_v5','_raw_vs_vanilla_rollback_v1')
    notebook['cells'][2]['source']=config.splitlines(keepends=True)
    notebook['cells'][5]['source']='''from experiments.shared_protocol import atomic_json_save
status_path = OUTPUT / 'arm_status.json'
statuses = json.loads(status_path.read_text()) if status_path.exists() else {}
comparison = {}
expected_hash = sha256_file(FORK)
for method in ARMS:
    status = statuses.get(method, {})
    if status.get('status') != 'completed':
        comparison[method] = {'available': False, 'status': status.get('status', 'not_completed')}
        continue
    result = json.loads((OUTPUT / 'arms' / method / 'result.json').read_text())
    identity = result['run_identity']
    if (result['theta_best_hash'] != expected_hash or result['protocol'] != EXPECTED
            or identity['cp_config'] != json.loads(json.dumps(asdict(CP)))
            or identity['algorithm_patience'] != ALGORITHM_PATIENCE
            or identity['post_fork_epochs'] != POST_FORK_EPOCHS
            or result['history'][-1]['post_fork_epoch'] != POST_FORK_EPOCHS):
        raise ValueError('Seed5 comparison requires the same complete fork/protocol/config/horizon')
    comparison[method] = {'available': True, **{key: result[key] for key in (
        'fork_epoch', 'historical_best_accuracy', 'report_best_accuracy',
        'report_best_loss', 'report_best_epoch', 'delta_vs_historical_best', 'scientific_escape')},
        'rollback_count': len(result['rollback_events']), 'fork_hash': expected_hash}
raw = comparison.get('e_driven_o_raw', {})
vanilla = comparison.get('vanilla_rollback', {})
eligible = raw.get('available', False) and vanilla.get('available', False)
report = {'arms': comparison, 'comparison_eligible': eligible,
    'e2o_minus_vanilla_rollback_best_accuracy':
        raw['report_best_accuracy'] - vanilla['report_best_accuracy'] if eligible else None}
atomic_json_save(report, OUTPUT / 'raw_vs_vanilla_rollback_comparison.json')
print(json.dumps(report, indent=2))
print('Status:', statuses)
'''.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
    path=Path('notebooks/kaggle_deit_tiny_seed5_raw_vs_vanilla_rollback.ipynb')
    path.write_text(json.dumps(notebook,indent=1)+'\n')
    return path

if __name__=='__main__':print(build())
