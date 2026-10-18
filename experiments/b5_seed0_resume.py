"""Select existing B5 trajectories without changing their code identity."""
import json
from pathlib import Path
import pickle
import subprocess

from experiments.stage_b5 import recipe, write_json
from scripts.collect_parallel import validate_result

LEGACY_COMMITS = {
    '8ea105fcf3194df48807e5c0d315272e8f1f6fd1',
    'd461ce96816ce6ae7e1bbdde2c78d5bed7cc20ab',
    'f571c6a7f39b6490f880b9ed4bd459741226dbcf',
    '3c41ecf87449b73ebff730c9be600f00818b1d24',
    '4d3008ad031663c93846d27003c9f7935b3f00a2',
}
GRADMAX_FIXED = {'3c41ecf87449b73ebff730c9be600f00818b1d24',
                '4d3008ad031663c93846d27003c9f7935b3f00a2'}


def select_sources(input_root, outputs, jobs, current_tag, current_commit, seed=0):
    if seed not in (0, 1, 2): raise ValueError('B5 seeds must be 0, 1 or 2')
    outputs = Path(outputs)
    allowed = LEGACY_COMMITS | {current_commit}

    def check(identity, job):
        commit = identity.get('code_commit')
        cfg = identity.get('cfg', {})
        if commit not in allowed or identity.get('source_digest') is not None:
            raise ValueError(f'{job}: unsupported checkpoint source {commit}')
        expected_rate = (float(job.split('__rate')[1].split('__')[0]) if '__rate' in job else
                         .001 if job.startswith('b5_cbp') else 0.)
        if (identity.get('setting') != 'regression' or identity.get('seed') != seed or
                identity.get('arm') != job.split('__')[0] or
                identity.get('algorithm_version') != 'v3' or
                identity.get('synthetic') is not False or identity.get('preflight_only') or
                identity.get('coefficient') != 1. or
                identity.get('upstream_commit') != 'a6b79580d85f3025bdb601566d3627c5f489f13b' or
                identity.get('expected_tasks') != 100 or
                int(cfg.get('num_data_points', 0)) != 1_000_000 or
                int(cfg.get('flip_after', 0)) != 10_000 or
                float(cfg.get('replacement_rate', -1)) != expected_rate):
            raise ValueError(f'{job}: incompatible B5 seed-{seed} identity')
        expected_cfg, _ = recipe('regression', 'cbp' if job.startswith('b5_cbp') else 'bp',
                                 expected_rate if job.startswith('b5_cbp') else None)
        expected_cfg.update(run_idx=0, use_gpu=0)
        if {k: v for k, v in cfg.items() if k not in ('data_file', 'env_file')} != expected_cfg:
            raise ValueError(f'{job}: original training recipe differs from the registered B5 setup')
        return commit

    # A standalone resume.pkl carries the original output path in its identity.
    for path in sorted(Path(input_root).rglob('resume.pkl')):
        if 'outputs' in path.relative_to(input_root).parts: continue
        with path.open('rb') as stream: state = pickle.load(stream)
        identity = state.get('identity', {})
        if identity.get('setting') != 'regression' or identity.get('seed') != seed: continue
        parts = Path(identity.get('cfg', {}).get('data_file', '')).parts
        if 'outputs' not in parts: raise ValueError(f'{path}: original output path missing')
        relative = parts[parts.index('outputs')+1:-1]
        if len(relative) == 2 and relative[1] not in jobs: continue
        known_tags = ('vna_v3_pilot', f'vna_v3_b5_seed{seed}', f'vna_v3_b5_seed{seed}_resume', current_tag)
        if len(relative) != 2 or relative[1] not in jobs or relative[0] not in known_tags:
            raise ValueError(f'{path}: unsupported original job directory')
        job = relative[1]
        check(identity, job)
        target = outputs.joinpath(*relative)
        destination = target/'resume.pkl'
        if destination.exists():
            if destination.read_bytes() != path.read_bytes():
                raise ValueError(f'{job}: conflicting standalone checkpoints; attach one snapshot')
        else:
            target.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.pkl.tmp')
            temporary.write_bytes(path.read_bytes()); temporary.replace(destination)
            write_json(target/'manifest.json', identity)
            print(f'IMPORTED {job}: step={state["step"]} source={path}', flush=True)

    selected = {}
    for job in jobs:
        candidates = []
        for tag in sorted(outputs.iterdir()):
            if not tag.is_dir(): continue
            out = tag/job
            checkpoint = out/'resume.pkl'
            summary_path = out/'summary.json'
            if not checkpoint.exists() and not summary_path.exists(): continue
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            state = None
            if checkpoint.exists():
                with checkpoint.open('rb') as stream: state = pickle.load(stream)
                identity = state['identity']
            else: identity = json.loads((out/'protocol.json').read_text())
            commit = check(identity, job)
            if job.startswith('b5_cbp_gradmax') and commit not in GRADMAX_FIXED | {current_commit}:
                print(f'RESTART {job}: legacy GradMax requires the numerical repair ({commit})', flush=True)
                continue
            done = summary.get('status') == 'COMPLETED'
            if done:
                tasks = json.loads((out/'tasks.json').read_text())
                protocol = json.loads((out/'protocol.json').read_text())
                validate_result(job, summary, protocol, tasks, commit)
                if protocol.get('cfg') != identity.get('cfg'):
                    raise ValueError(f'{job}: checkpoint/result configuration mismatch')
            step = int(state['step']) if state else int(summary.get('steps', 1_000_000))
            if not 0 <= step <= 1_000_000 or (state and state['results']['count'] != step):
                raise ValueError(f'{job}: invalid checkpoint step/count')
            if not done and commit != current_commit and job not in (f'b5_bp__s{seed}', f'b5_cbp__s{seed}') and '__rate' not in job:
                schedule = identity.get('preflight')
                if not schedule or schedule.get('code_commit') != commit or schedule.get('source_digest') is not None:
                    raise ValueError(f'{job}: checkpoint lacks its original authority schedule')
                schedule_path = tag/'_schedules/regression.json'
                if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule:
                    raise ValueError(f'{job}: conflicting original authority schedule')
                write_json(schedule_path, schedule)
            if state and not (out/'tasks.json').exists():
                from adapters.lop_hooks import StepLog
                results = StepLog(1_000_000, 10_000, 1, False)
                results.restore(state['results'])
                metrics = {r['task']: r for r in (state.get('intervention') or {}).get('task_metrics', [])}
                write_json(out/'tasks.json', [{**r, **metrics.get(r['task'], {})} for r in results.tasks()])
            candidates.append(dict(out=out, commit=commit, step=step, completed=done))
        if candidates:
            source = max(candidates, key=lambda r: (r['completed'], r['step'], r['commit'] == current_commit))
            selected[job] = source
            action = 'SKIP_COMPLETED' if source['completed'] else 'FINALIZE_CHECKPOINT' if source['step'] == 1_000_000 else 'RESUME'
            print(f'{action} {job}: step={source["step"]}/1000000 '
                  f'commit={source["commit"]} path={source["out"]}', flush=True)
        else:
            print(f'NEW {job}: no compatible checkpoint/result found', flush=True)
    if not selected and seed == 0:
        raise FileNotFoundError('No reusable B5 seed-0 checkpoint/result found; attach the old Output/archive or resume.pkl before running')
    write_json(outputs/current_tag/'resume_sources.json',
               {job: {**source, 'out': str(source['out'])} for job, source in selected.items()})
    return selected


def prepare_seed_schedule(input_root, outputs, run_tag, current_commit):
    """Reuse a seed-0 authority schedule and its exact source for fresh jobs."""
    outputs = Path(outputs)
    destination = outputs/run_tag/'_schedules/regression.json'
    def read(path):
        value = json.loads(path.read_text())
        if (value.get('setting') != 'regression' or value.get('authority_arm') != 'b5_bp_vna_gir' or
                value.get('authoritative') is not True or value.get('synthetic') is not False or
                value.get('source_digest') is not None or
                value.get('code_commit') not in GRADMAX_FIXED | {current_commit} or
                type(value.get('K_base')) is not int or value['K_base'] < 1):
            raise ValueError(f'incompatible shared B5 schedule: {path}')
        return value
    if destination.exists():
        schedule = read(destination)
    else:
        candidates = []
        for root in (Path(input_root), outputs):
            for path in sorted(root.rglob('regression.json')):
                try: candidates.append((path, read(path)))
                except (OSError, ValueError): continue
        if not candidates:
            raise FileNotFoundError('Attach seed-0 Output/archive containing the repaired B5 regression.json; seeds 1/2 reuse that schedule without new preflight')
        preferred = [current_commit, '4d3008ad031663c93846d27003c9f7935b3f00a2',
                     '3c41ecf87449b73ebff730c9be600f00818b1d24']
        commit = next(c for c in preferred if any(v['code_commit'] == c for _, v in candidates))
        same = [(p, v) for p, v in candidates if v['code_commit'] == commit]
        if any(value != same[0][1] for _, value in same):
            raise ValueError('Conflicting B5 authority schedules; attach one seed-0 calibration snapshot')
        path, schedule = same[0]
        write_json(destination, schedule)
        print(f'SHARED_SCHEDULE source={path}', flush=True)
    print(f'SHARED_SCHEDULE K_base={schedule["K_base"]} commit={schedule["code_commit"]}; no new preflight', flush=True)
    return schedule['code_commit']


def checkout_sources(repo, sources):
    repo = Path(repo)
    result = {}
    for commit in sorted({r['commit'] for r in sources.values() if not r['completed']}):
        if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == commit:
            result[commit] = repo
            continue
        target = repo.parent/f'{repo.name}-b5-{commit[:8]}'
        if not target.exists():
            subprocess.run(['git', 'clone', '--no-checkout', str(repo), str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', commit], check=True)
        if subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip() != commit:
            raise ValueError(f'wrong B5 source checkout: {target}')
        upstream = target/'third_party/loss-of-plasticity'
        if not upstream.exists(): upstream.symlink_to(repo/'third_party/loss-of-plasticity', target_is_directory=True)
        result[commit] = target
    return result


RESUME_ENTRY = '''import runpy
from adapters.lop_hooks import LearnHook
original = LearnHook._restore_resume
def restore(self, learner, frame):
    result = original(self, learner, frame)
    print(f'RESUME_APPLIED step={self.step}: model, optimizer, RNG and hook state restored', flush=True)
    return result
LearnHook._restore_resume = restore
runpy.run_module('experiments.pilot_v3', run_name='__main__')
'''
