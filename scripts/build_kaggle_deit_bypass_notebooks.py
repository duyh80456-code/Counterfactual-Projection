"""One fresh-seed E→O raw versus paper relaxed Bypass notebook per seed."""
import copy
import json
from pathlib import Path


def build(seed):
    notebook=copy.deepcopy(json.loads(Path('notebooks/kaggle_deit_tiny_seed2_rollback_comparison.ipynb').read_text()))
    notebook['cells'][0]['source']=f'''# DeiT-Tiny seed {seed}: E→O raw top1 vs relaxed GELU Bypass

Attach CIFAR-100, enable Internet, T4x2 and Kaggle Secret `github_token`.
Optional: full previous output archive from this exact seed/config for resume.
This seed trains its own Vanilla and must not reuse another seed's fork.

Phase1: train Vanilla until150 epochs without a strict validation-accuracy best.
No minimum epoch. Reload the historical-best model/AdamW/scheduler/RNG/loader.
Phase2: GPU0 E→O raw top1, GPU1 `deit_bypass`;150 training epochs each.
No A3/A4 or additional arm. Original raw TINY WHERE over12 sites, rank4,
WHERE3x32, projection32, gate32, adaptive CG200, scales(.0125,.025,.05).
E→O uses patience10 rollback (accuracy/lower loss tie anchor), current RNG/data
stream, and no retrigger. Report best uses strict accuracy with paired loss.

Bypass follows Jung/Lee's relaxed construction (Eq.8 / Section IV-A):
`GELU(z) + d ⊙ z` at all12 original MLP activations, initialized at d=0.
It trains all original weights and D: opt1=100 epochs; opt2 has at most50 epochs,
with task loss + `(3e-6 * opt2_step) * sum(||d||₂)`.
When summed contraction norm <.002, drop D and spend remaining epochs in train3.
No forced projection, rollback or scheduler rebase in Bypass. The existing
AdamW moments/groups are preserved; new D vectors join the no-decay group.
This is a GELU adaptation with a matched budget, not a reproduction of the
paper's original network/hyperparameter schedule.

Only original-space validation observations after contraction qualify as Bypass
accuracy comparisons. Expanded-space best is a separate diagnostic. If opt2
exhausts the budget before contraction, `accuracy_comparison_eligible=false`
and the comparison delta is null. Validation never controls contraction.

Recipe: batch64, AdamW LR5e-4/WD.05, warmup5, cosine schedule300, max_epoch800,
validation3000, plateau150 and post-fork150. Schedule300 is not a minimum epoch.
Full latest saves every epoch, including Bypass phase/counters/D/optimizer/RNG,
or E controller/anchor. Completed arms skip training and partial arms resume.
Outputs: result/checkpoints/epoch logs and bypass_comparison.csv/json. Download
this notebook's full output tar.gz for the next Kaggle session.
Paper: https://www.donghunlee.com/papers/Jung_Lee_Bypass__IEEE_TNNLS.pdf
'''.splitlines(keepends=True)
    bootstrap=''.join(notebook['cells'][1]['source'])
    bootstrap=bootstrap.replace('"tests/test_deit_resume.py"','"tests/test_deit_resume.py", "tests/test_deit_bypass.py"')
    notebook['cells'][1]['source']=bootstrap.splitlines(keepends=True)
    config=''.join(notebook['cells'][2]['source']).replace('SEED = 2',f'SEED = {seed}')
    config=config.replace("ARMS = ('vanilla_rollback', 'o_projection_only', 'o_projection_only_rollback', 'e_driven_o_raw')",
        "ARMS = ('e_driven_o_raw', 'deit_bypass')")
    config=config.replace('_rollback_comparison_v5','_raw_vs_bypass_v1')
    config += '\nBYPASS_OPT1_EPOCHS = 100\nBYPASS_MAX_OPT2_EPOCHS = 50\n'
    notebook['cells'][2]['source']=config.splitlines(keepends=True)
    command=''.join(notebook['cells'][4]['source'])
    command=command.replace("'--arms', ','.join(ARMS)]","'--arms', ','.join(ARMS),\n'--bypass-opt1-epochs', str(BYPASS_OPT1_EPOCHS), '--bypass-max-opt2-epochs', str(BYPASS_MAX_OPT2_EPOCHS)]")
    notebook['cells'][4]['source']=command.splitlines(keepends=True)
    notebook['cells'][5]['source']='''path = OUTPUT / 'bypass_comparison.json'
if path.exists():
    print(json.dumps(json.loads(path.read_text()), indent=2))
print((OUTPUT / 'arm_status.json').read_text())
'''.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
    path=Path(f'notebooks/kaggle_deit_tiny_seed{seed}_raw_vs_bypass.ipynb')
    path.write_text(json.dumps(notebook,indent=1)+'\n')
    return path

if __name__=='__main__':
    for seed in (3,4,5):print(build(seed))
