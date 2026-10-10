"""Strict fork discovery/resume for new seed2 arms; old runs are read-only comparisons."""
from pathlib import Path
import json
import os
from dataclasses import asdict
import shutil
import tarfile
import zipfile
import torch
from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.deit_resume import unique, FULL_STATE
from experiments.deit_protocol import checked_source
from experiments.run_deit_projection_aware import identity_for, finish, validate_seed2_fork
from adapters.deit_ablation import res18_cp_config
from experiments.shared_protocol import sha256_file, atomic_json_save

ARMS = ('e2o_top3_best_gate', 'e2o_top3_recurrent')


def prepare_projection_resume(roots, output):
    output = Path(output); states = []; rejected = []
    for index, root in enumerate(dict.fromkeys(map(str, [output, *roots]))):
        if not Path(root).exists():
            continue
        found, failures = discover_checkpoints(root, output / '_input_cache' / str(index),
            kind={'deit_plateau_fork', 'deit_projection_aware_latest'},
            payload_filter=lambda p: p.get('protocol', {}).get('seed') == 2
                and p.get('protocol', {}).get('protocol_version') == 5 and FULL_STATE.issubset(p),
            payload_transform=lambda p: {k: v for k, v in p.items() if k not in FULL_STATE
                and k not in ('e_controller', 'previous_direction')})
        states.extend(found); rejected.extend(failures)
    fork_item = unique([item for item in states if item['payload']['kind'] == 'deit_plateau_fork'
        and item['payload']['epoch'] == 375 and abs(item['payload']['historical_best_accuracy'] - .562) < 1e-12], 'seed2 fork375')
    if not fork_item:
        atomic_json_save({'rejected': rejected}, output / 'resume_plan.json')
        raise FileNotFoundError('Attach seed2 plateau_checkpoint.pt: historical-best epoch375 /56.20%; no old arms will rerun')
    fork_path = output / 'plateau_checkpoint.pt'
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            temporary = target.with_suffix('.importing')
            try:
                shutil.copy2(source, temporary); temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    copy(fork_item['path'], fork_path)
    fork, recipe = checked_source(fork_path, {'deit_plateau_fork'})
    validate_seed2_fork(fork, recipe)
    fork_hash = sha256_file(fork_path)
    plan = {'fork_path': str(fork_path), 'fork_hash': fork_hash, 'fork_epoch': 375,
            'historical_best_accuracy': .562, 'arms': {}, 'rejected': rejected}
    for method in ARMS:
        expected = identity_for(fork_hash, method, res18_cp_config(), 150, 10)
        candidates = [item for item in states if item['payload']['kind'] == 'deit_projection_aware_latest'
                      and item['payload']['run_identity'].get('method') == method]
        if any(item['payload'].get('run_identity') != expected or item['payload']['protocol'] != fork['protocol'] for item in candidates):
            raise ValueError(f'{method}: conflicting fork/config/seed; use a separate output')
        selected = None
        if candidates:
            progress = lambda item: (item['payload']['completed_epochs'], item['payload']['search_count'],
                                     not item['payload']['pending_search'])
            maximum = max(map(progress, candidates))
            selected = unique([item for item in candidates if progress(item) == maximum], method)
        if selected:
            target = output / method / 'checkpoint_latest.pt'
            copy(selected['path'], target)
            state = torch.load(target, map_location='cpu', weights_only=False)
            from experiments.deit_e_rollback import EAccuracyRollback
            count = state['completed_epochs']
            if (not 0 <= count <= 150 or state['epoch'] != 375 + count
                    or len(state['history']) != count + 1 or state['search_count'] != len(state['interventions'])):
                raise ValueError('inconsistent imported arm progress')
            EAccuracyRollback.from_state(state['e_controller'], 10, count, stall_on_anchor=True)
            action = 'completed' if count == 150 and not state['pending_search'] else 'resume'
            if action == 'completed':
                finish(state, target.parent)
            plan['arms'][method] = {'action': action, 'completed_epochs': count, 'pending_search': state['pending_search']}
        else:
            plan['arms'][method] = {'action': 'start', 'completed_epochs': 0}
    atomic_json_save(plan, output / 'resume_plan.json')
    print('Projection-aware resume:', {key: value for key, value in plan.items() if key != 'rejected'}, flush=True)
    return plan



def iter_results(roots):
    # Old result files can live in an output archive, not only expanded mounts.
    for root in roots:
        for directory, _, files in os.walk(root, followlinks=True):
            for filename in files:
                path = Path(directory) / filename
                try:
                    if filename == 'result.json':
                        yield str(path), json.loads(path.read_text())
                    elif filename.endswith('.zip'):
                        with zipfile.ZipFile(path) as archive:
                            for info in archive.infolist():
                                if Path(info.filename).name == 'result.json' and info.file_size < 32 * 1024**2:
                                    yield str(path) + '::' + info.filename, json.loads(archive.read(info))
                    elif filename.endswith(('.tar', '.tar.gz', '.tgz')):
                        with tarfile.open(path, 'r|*') as archive:
                            for info in archive:
                                if info.isfile() and Path(info.name).name == 'result.json' and info.size < 32 * 1024**2:
                                    yield str(path) + '::' + info.name, json.load(archive.extractfile(info))
                except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile, EOFError):
                    continue


def comparison_table(roots, output, fork_hash):
    output = Path(output)
    fork, _ = checked_source(output / 'plateau_checkpoint.pt', {'deit_plateau_fork'})
    from experiments.run_deit_projection_aware import write_csv
    names = {'e_driven_o_raw', 'vanilla_rollback', 'o_projection_only_rollback', *ARMS}
    matches = {}; excluded = []; seen = set()
    for path, result in iter_results([output, *map(Path, roots)]):
        if path in seen:
            continue
        seen.add(path)
        method = result.get('method')
        if method not in names:
            continue
        identity = result.get('run_identity', {})
        history = result.get('history', [])
        completed = result.get('completed_epochs', history[-1].get('post_fork_epoch') if history else None)
        if (result.get('theta_best_hash', identity.get('fork_hash')) != fork_hash
                or result.get('protocol') != fork['protocol']
                or identity.get('cp_config') != json.loads(json.dumps(asdict(res18_cp_config())))
                or identity.get('post_fork_epochs') != 150 or identity.get('algorithm_patience') != 10
                or completed != 150):
            excluded.append({'path': str(path), 'method': method, 'reason': 'fork/protocol/config/horizon mismatch'})
            continue
        if method in matches and matches[method] != result:
            excluded.append({'path': str(path), 'method': method, 'reason': 'multiple distinct matching runs; not cherry-picking'})
            matches[method] = None
        elif method not in matches:
            matches[method] = result
    rows = []
    for method in ('e_driven_o_raw', *ARMS, 'vanilla_rollback', 'o_projection_only_rollback'):
        result = matches.get(method)
        if result is None:
            rows.append({'method': method, 'available': False}); continue
        interventions = result.get('interventions', [])
        rows.append({'method': method, 'available': True, 'report_best_accuracy': result['report_best_accuracy'],
            'report_best_loss': result['report_best_loss'], 'report_best_epoch': result['report_best_epoch'],
            'delta_vs_historical_best': result['report_best_accuracy'] - .562,
            'proposals_tried': result.get('proposals_tried'),
            'corrections_accepted': result.get('corrections_accepted', sum(r.get('correction_applied', False) for r in interventions)),
            'rollback_count': len(result.get('rollback_events', [])), 'search_rounds': result.get('search_rounds', len(interventions)),
            'search_seconds': result.get('search_seconds'), 'source_run_identity': result['run_identity']})
    best = {row['method']: row['report_best_accuracy'] for row in rows if row['available']}
    effects = {}
    for label, a, b in [('selection', ARMS[0], 'e_driven_o_raw'), ('recurrent', ARMS[1], ARMS[0]),
                        ('structural_A3', ARMS[0], 'o_projection_only_rollback'), ('structural_A4', ARMS[1], 'o_projection_only_rollback')]:
        effects[label] = best[a] - best[b] if a in best and b in best else None
    write_csv(rows, output / 'comparison.csv')
    atomic_json_save({'rows': rows, 'effects': effects, 'excluded': excluded,
        'compute_budget_caveat': 'Top3 and recurrent searches spend more proposals/CG than old top1'}, output / 'comparison.json')
    return rows
