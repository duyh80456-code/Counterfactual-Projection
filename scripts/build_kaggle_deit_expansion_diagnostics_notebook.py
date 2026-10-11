"""A short seed2 diagnostic notebook; no A3/A4 or post-fork training."""
import copy
import json
from pathlib import Path


def build():
    notebook=copy.deepcopy(json.loads(Path('notebooks/kaggle_deit_tiny_seed2_projection_aware.ipynb').read_text()))
    notebook['cells'][0]['source']='''# DeiT seed2 — E0/E1/E2/E3/E4 expansion-generator diagnostics

Attach CIFAR-100 and the original full seed2 fork: epoch375 / validation56.20%.
Attach the previous D0 output archive/result if available; it must match fork hash,
protocol and CP config. Without it, D0 all12 is recomputed first and explicitly
labelled a fresh reference, not verification against a missing old artifact.
Enable Internet, a GPU, and `github_token`. This notebook does not train A3/A4,
run150 epochs, apply a correction, or update the original optimizer.

E0 reproduces all12 native rank4 TINY WHERE gains/norm/rank before proceeding.
Default selected sites: TINY top1, smallest D0 projection residual, median TINY
rank; deterministic fillers produce three distinct sites if selections overlap.
E3/E4 run first for all selected sites, followed by E1 ranks(2,4,8,16) and E2
probe epsilons(.01,.05,.1,.2) × expansion scales(.25,.5,1,2). Duplicate cases share
one proposal/solve. Set ALL_BLOCKS=True to deliberately extend the suite to12;
there is no automatic selection based on validation or a new acceptance threshold.

Generators: original TINY; temporary A/a/B optimized on disjoint opt_fit64 for20
Adam steps (A nonzero, B=0 initially); random auxiliary neurons normalized to the
TINY projection-logit norm. Failed/degenerate norm matching is labelled. No
comparison is claimed to match compute; the unique-proposal budget records
forward/backward counts, wall times and seeds.

`probe_epsilon` is the existing virtual expansion measurement gate; native TINY
proposal generation depends on rank/statistics, not this epsilon. Expansion scale
is a separate control. Default exact_expanded_forward reruns the entire expanded
network with gate=epsilon*scale; linear_logits separately implements scale*dE.
Gradient-tuned branches are trained/evaluated at gate1; this difference is logged.

FP32/FP64 CE metrics use the same original forward logits. Default additionally
recomputes the whole frozen model and fixed proposal in FP64; this does not
recover errors incurred while constructing the original FP32 TINY proposal.
Validation is never evaluated or used to choose a method/hyperparameter. WHERE,
projection, gate and tuning roles remain distinct; gate is a selection diagnostic.
Every projection affects only the original MLP of the expansion's own site and
is previewed without commit. Model/Parameter/optimizer/RNG invariance is checked.

Outputs: expansion_diagnostics.csv, existing_tiny_diagnostics.csv, configuration,
summary, unique-proposal budget, actual probe tensors/indices, cached proposals
and diagnostic_progress.pt. Resume skips committed diagnostic rows, not training
arms. Download the archive for another Kaggle session.
'''.splitlines(keepends=True)
    bootstrap=''.join(notebook['cells'][1]['source']).replace('"tests/test_deit_projection_aware.py"',
        '"tests/test_deit_projection_aware.py", "tests/test_deit_expansion_diagnostics.py"')
    notebook['cells'][1]['source']=bootstrap.splitlines(keepends=True)
    config=''.join(notebook['cells'][2]['source']).replace('deit_seed2_projection_aware_v1','deit_seed2_expansion_diagnostics_v1')
    config=config.replace("GPU_DEVICES = 'auto'\n",'')
    config+='\nALL_BLOCKS = False\nVERIFY_FORWARD_FP64 = True\nAUXILIARY_STEPS = 20\n'
    notebook['cells'][2]['source']=config.splitlines(keepends=True)
    notebook['cells'][3]['source']='''from experiments.run_deit_expansion_diagnostics import prepare_inputs
FORK, _, _, D0 = prepare_inputs(['/kaggle/input'], OUTPUT)
print('Fork:', FORK, 'Existing D0:', D0 is not None)
'''.splitlines(keepends=True)
    notebook['cells'][4]['source']='''command = [sys.executable, '-m', 'experiments.run_deit_expansion_diagnostics',
    '--data-root', str(DATA_ROOT), '--output', str(OUTPUT), '--input-root', '/kaggle/input',
    '--auxiliary-steps', str(AUXILIARY_STEPS)]
if ALL_BLOCKS: command += ['--all-blocks']
if not VERIFY_FORWARD_FP64: command += ['--no-forward-fp64']
try:
    subprocess.run(command, cwd=REPO, env=ENV, check=True)
finally:
    archive = shutil.make_archive(str(OUTPUT), 'gztar', root_dir=OUTPUT)
    print('Resumable diagnostics:', archive)
'''.splitlines(keepends=True)
    notebook['cells'][5]['source']='''print((OUTPUT / 'summary.json').read_text())
import pandas as pd
frame = pd.read_csv(OUTPUT / 'expansion_diagnostics.csv')
display(frame[['proposal_method','block_name','rank','probe_epsilon','expansion_scale',
    'e_gain_fp32','e_gain_fp64','e_gain_forward_fp64','projection_residual','projection_cosine','gate_gain','accepted']])
'''.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
    path=Path('notebooks/kaggle_deit_tiny_seed2_expansion_diagnostics.ipynb')
    path.write_text(json.dumps(notebook,indent=1)+'\n');return path

if __name__=='__main__':print(build())
