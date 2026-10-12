"""Load an existing R18 seed1 plateau; recurrent raw E and genuine Vanilla rollback."""
import copy
import json
from pathlib import Path


def build():
    base=json.loads(Path('notebooks/kaggle_plateau_fork_t4x2.ipynb').read_text())
    def cell(kind,text):
        result={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':result.update(execution_count=None,outputs=[])
        return result
    bootstrap=copy.deepcopy(base['cells'][1])
    bootstrap['source']=''.join(bootstrap['source']).replace(
        'plateau_fork_three_methods_150ep_v6','resnet18_seed1_raw_vs_vanilla_rollback_v1').splitlines(keepends=True)
    setup=''.join(base['cells'][2]['source']).replace(
        'subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO,\n               env=test_env, check=True)\n','')
    setup=setup[:setup.index('cifar_dirs = sorted')]+'''roots = set()
for directory, _, files in os.walk('/kaggle/input', followlinks=True):
    if Path(directory).name == 'cifar-100-python' and {'train', 'meta'}.issubset(files):
        roots.add(Path(directory).parent.resolve())
if len(roots) != 1:
    raise FileNotFoundError(f'Attach exactly one CIFAR-100 dataset: {roots}')
DATA_ROOT = next(iter(roots))
'''
    setup += "\nENV = test_env\nprint('GPU workers:', {i: torch.cuda.get_device_name(i) for i in (0, 1)})\n"
    cells=[cell('markdown', '''# ResNet18 seed1 — raw E→O versus Vanilla rollback

Attach CIFAR-100 and the existing seed1 ResNet18 `plateau_checkpoint.pt`, or its
full output tar/zip. Enable Internet, Kaggle T4x2 and secret `github_token`.
This notebook requires an existing validation-selected plateau; it never trains
Phase1 again. Other seeds/backbones and trigger-selected plateaus are excluded.

GPU0 runs original raw TINY E→O: initial all-eight-site WHERE, winner projection,
then stall → restore anchor → query E again. GPU1 trains genuine Vanilla from the
same full fork; stall only restores its anchor, without E/projection/momentum reset.
Both get150 SGD epochs and patience10. Report best is strict validation accuracy;
controller anchor uses accuracy, then lower loss on an exact tie. Rollback restores
model/SGD momentum/scheduler but keeps the live RNG and data stream.

Original R18 intervention settings remain rank4, statistics256, WHERE3×32,
projection32, gate32, scales(.0125,.025,.05), CG200. LR is inherited from the fork,
never restarted or overridden. Trigger metrics remain diagnostic; validation
selects anchors. Official test is unused. Full latest/best checkpoints, a final
comparison table and accuracy/loss rollback plots are saved in the output archive.
'''),bootstrap,cell('code',setup),cell('code','''if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.deit_resume import unique, most_advanced
from experiments.shared_protocol import sha256_file, atomic_json_save
from experiments.run_deit_all_arms import run_jobs_parallel

SEED = 1
HORIZON = 150
PATIENCE = 10
ARMS = ('ours_e_driven_o', 'vanilla_rollback')
FULL_STATE = {'model', 'optimizer', 'scheduler', 'rng', 'train_loader_generator_state',
              'train_indices', 'evaluation_indices', 'trigger_indices', 'source_tuning_indices'}
def compatible_fork(payload):
    protocol = payload.get('protocol', {})
    return (FULL_STATE.issubset(payload) and protocol.get('architecture') == 'CIFAR-ResNet18'
            and protocol.get('seed', payload.get('seed')) == SEED
            and 'validation' in protocol.get('selection_metric', '').lower())
forks = []
for index, root in enumerate((OUTPUT, Path('/kaggle/input'))):
    found, rejected = discover_checkpoints(root, OUTPUT / '_resume_cache' / f'fork_{index}',
        kind='plateau_fork_checkpoint', payload_filter=compatible_fork,
        payload_transform=lambda p: {'epoch': p['epoch'], 'protocol': p['protocol']})
    forks.extend(found)
    print('Excluded checkpoint count:', len(rejected))
selected = unique(forks, 'seed1 ResNet18 validation plateau')
if selected is None:
    raise FileNotFoundError('Attach a full validation-selected ResNet18 seed1 plateau with seed/architecture metadata.')
PLATEAU_CHECKPOINT = OUTPUT / 'plateau_checkpoint.pt'
if selected['path'].resolve() != PLATEAU_CHECKPOINT.resolve():
    shutil.copy2(selected['path'], PLATEAU_CHECKPOINT)
PLATEAU_HASH = sha256_file(PLATEAU_CHECKPOINT)
source = torch.load(PLATEAU_CHECKPOINT, map_location='cpu', weights_only=False)
FORK_EPOCH = int(source['epoch'])
print('Loaded fork:', {'seed': SEED, 'architecture': source['protocol']['architecture'],
    'epoch': FORK_EPOCH, 'sha256': PLATEAU_HASH,
    'learning_rates': [group['lr'] for group in source['optimizer']['param_groups']],
    'model_optimizer_scheduler_rng_loader': 'preserved'})
del source

CONFIG = dict(batch_size=64, workers=2, weight_decay=5e-4, rank=4, probe_epsilon=.05,
    statistics_samples=256, where_batches=3, where_samples=32, projection_samples=32,
    gate_samples=32, cg_iterations=200, cg_relative_tolerance=.01,
    cg_preconditioner_probes=8, damping=.001, line_search_scales='0.0125,0.025,0.05',
    site='stages.2.blocks.0', site_selection_mode='all_functional_gain')
def compatible_arm(payload):
    protocol = payload.get('protocol', {})
    method = protocol.get('method')
    mode = {'ours_e_driven_o': 'recurrent_best_rollback', 'vanilla_rollback': 'rollback_only'}.get(method)
    schedule = protocol.get('intervention_schedule') or {}
    return (FULL_STATE.issubset(payload) and payload.get('theta_best_hash') == PLATEAU_HASH
        and protocol.get('architecture') == 'CIFAR-ResNet18' and protocol.get('seed') == SEED
        and protocol.get('post_fork_epochs') == HORIZON and mode is not None
        and schedule.get('mode') == mode and schedule.get('patience') == PATIENCE
        and protocol.get('continuation_config') == CONFIG)
progress = []
for index, root in enumerate((OUTPUT, Path('/kaggle/input'))):
    found, _ = discover_checkpoints(root, OUTPUT / '_resume_cache' / f'arms_{index}',
        kind={'plateau_fork_arm_progress', 'plateau_fork_arm_best'}, payload_filter=compatible_arm,
        payload_transform=lambda p: {k: p[k] for k in ('kind', 'post_fork_epoch', 'protocol')})
    progress.extend(found)
for method in ARMS:
    matching = [item for item in progress if item['payload']['protocol']['method'] == method]
    for kind, filename in (('plateau_fork_arm_progress', 'checkpoint_latest.pt'),
                           ('plateau_fork_arm_best', 'checkpoint_best.pt')):
        chosen = most_advanced([item for item in matching if item['payload']['kind'] == kind],
                               'post_fork_epoch', method + ' ' + kind)
        if chosen:
            destination = OUTPUT / method / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if chosen['path'].resolve() != destination.resolve():
                shutil.copy2(chosen['path'], destination)
            print('Resume:', method, filename, chosen['payload']['post_fork_epoch'])
'''),cell('code','''commands = {}
for method in ARMS:
    command = [sys.executable, '-m', 'experiments.run_plateau_fork', '--method', method,
        '--reference-root', str(REFERENCE), '--data-root', str(DATA_ROOT),
        '--plateau-checkpoint', str(PLATEAU_CHECKPOINT), '--plateau-checkpoint-hash', PLATEAU_HASH,
        '--output', str(OUTPUT / method), '--architecture', 'resnet18', '--seed', str(SEED),
        '--post-fork-epochs', str(HORIZON), '--retrigger-patience', str(PATIENCE),
        '--record-continuation-config']
    for key, value in CONFIG.items():
        command += ['--' + key.replace('_', '-'), str(value)]
    commands[method] = command
try:
    statuses = run_jobs_parallel(commands, OUTPUT / 'arm_status.json', [0, 1], parent_env=ENV)
finally:
    print('Resumable archive:', shutil.make_archive(str(OUTPUT), 'gztar', root_dir=OUTPUT))
print('Arm statuses:', statuses)
'''),cell('code','''import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

results, rows = {}, []
for method in ARMS:
    status = statuses.get(method, {}).get('status', 'not_started')
    row = {'method': method, 'status': status}
    path = OUTPUT / method / 'result.json'
    if status == 'completed' and path.exists():
        result = json.loads(path.read_text())
        if (result['theta_best_hash'] != PLATEAU_HASH or result['post_fork_epochs'] != HORIZON
                or result['protocol']['continuation_config'] != CONFIG):
            raise ValueError('Comparison fork/config/horizon mismatch')
        results[method] = result
        row.update(best_acc_pct=100 * result['report_best_accuracy'], best_loss=result['report_best_loss'],
            best_epoch=result['report_best_epoch'], rollbacks=result['rollback_count'],
            interventions=result['intervention_count'], scientific_escape=result['scientific_escape'],
            delta_vs_fork_pp=100 * result['best_validation_accuracy_delta'])
    elif status == 'completed':
        row['status'] = 'missing_result'
    rows.append(row)
vanilla = results.get('vanilla_rollback')
for row in rows:
    row['delta_vs_vanilla_pp'] = (row['best_acc_pct'] - 100 * vanilla['report_best_accuracy']
        if row['status'] == 'completed' and vanilla and 'best_acc_pct' in row else None)
table = pd.DataFrame(rows)
display(table)
table.to_csv(OUTPUT / 'comparison.csv', index=False)
atomic_json_save({'fork_hash': PLATEAU_HASH, 'rows': rows}, OUTPUT / 'comparison.json')
plots = OUTPUT / 'plots'; plots.mkdir(exist_ok=True)
for method, result in results.items():
    history = pd.DataFrame(result['history'])
    fig, axes = plt.subplots(1, 2, figsize=(15, 4), constrained_layout=True)
    for ax, observed, anchor, factor in ((axes[0], 'validation_accuracy', 'controller_anchor_accuracy', 100),
                                        (axes[1], 'validation_loss', 'controller_anchor_loss', 1)):
        ax.plot(history['post_fork_epoch'], factor * history[observed], label='Before controller')
        ax.step(history['post_fork_epoch'], factor * history[anchor], where='post', linestyle='--', label='Anchor after controller')
        for event in result['rollback_events']:
            x = event['post_fork_epoch']; y = factor * event['anchor_accuracy' if factor == 100 else 'anchor_loss']
            ax.axvline(x, color='red', linestyle=':', alpha=.3)
            ax.scatter([x], [y], marker='v', color='red', s=25)
        ax.set(xlabel='Elapsed post-fork epochs', ylabel=observed + (' (%)' if factor == 100 else ''))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(method + ' — rollback markers are restored anchors, not fresh validation')
    fig.savefig(plots / (method + '.png'), dpi=180)
    fig.savefig(plots / (method + '.pdf'))
    plt.show(); plt.close(fig)
print('Resumable archive:', shutil.make_archive(str(OUTPUT), 'gztar', root_dir=OUTPUT))
''')]
    notebook={'cells':cells,'metadata':copy.deepcopy(base['metadata']),'nbformat':4,'nbformat_minor':5}
    path=Path('notebooks/kaggle_resnet18_seed1_raw_vs_vanilla_rollback.ipynb')
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
    return path


if __name__=='__main__':print(build())
