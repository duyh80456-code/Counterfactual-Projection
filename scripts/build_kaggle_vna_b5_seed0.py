"""Build a full B5 notebook for one seed using the shared resumable runner."""
import argparse
import json
from pathlib import Path

from scripts.build_kaggle_vna_parallel import build, PRESETS
from scripts.build_kaggle_vna_signal_curve_notebook import cell

B5_GROUPS = {
    'A': ('bp_kself', 'cbp_gir', 'bp_gir', 'bp'),
    'B': ('cbp_vna_gir', 'bp_kself', 'bp_vna_gir', 'cbp'),
    'C': ('cbp_gradmax', 'bp_vna', 'bp_vna', 'bp_vna_rand'),
}


PROGRESS = r'''
from scripts.collect_parallel import validate_result
from experiments.b5_seed0_resume import select_sources, checkout_sources, prepare_seed_schedule, RESUME_ENTRY
seed0_failures = {}
seed0_sources = {}

def seed0_out(job):
    return seed0_sources.get(job, {}).get('out', STAGE_B / RUN_TAG / job)

def seed0_progress():
    active = {lane.get('job') for lane in list(lane_state.values())}
    rows = []
    for job in JOBS_CPU:
        out = seed0_out(job)
        source_commit = seed0_sources.get(job, {}).get('commit', REPO_COMMIT)
        def read(name, default):
            try: return json.loads((out / name).read_text())
            except (OSError, ValueError): return default
        tasks = read('tasks.json', [])
        summary = read('summary.json', {})
        status = read('status.json', {})
        protocol = read('protocol.json', {})
        expected = protocol.get('expected_tasks', 100)
        done = summary.get('status') == 'COMPLETED'
        if done:
            validate_result(job, summary, protocol, tasks, source_commit, SOURCE_DIGEST)
        state = ('FAILED' if job in seed0_failures else 'COMPLETED' if done else
                 'RUNNING' if job in active else
                 'PAUSED' if session_paused.is_set() and (out / 'resume.pkl').exists() else
                 'WAITING_FINALIZE' if seed0_sources.get(job, {}).get('step') == 1000000 else
                 'WAITING_RESUME' if (out / 'resume.pkl').exists() else 'PENDING')
        rows.append(dict(job=job, status=state, tasks=sum(r.get('step', 0) % 10000 == 0 for r in tasks), total=expected,
            step=status.get('step', seed0_sources.get(job, {}).get('step')),
            source_commit=source_commit[:7], source_tag=out.parent.name,
            online_error=tasks[-1].get('online_error') if tasks else None,
            mean_last50=summary.get('mean_last50') if done else None,
            error=seed0_failures.get(job, '')))
    import pandas as pd
    table = pd.DataFrame(rows)
    print('\nSEED 0: full horizon, lower error is better; partial errors are not final scores.', flush=True)
    print(table.to_string(index=False), flush=True)
    print(f"Completed {sum(r['status']=='COMPLETED' for r in rows)}/{len(rows)} jobs; "
          f"failed {len(seed0_failures)}", flush=True)
    return table

def seed0_results():
    table = seed0_progress()
    table.to_csv(STAGE_B / RUN_TAG / 'seed0_progress.csv', index=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for job in JOBS_CPU:
        out = seed0_out(job)
        if not (out / 'tasks.json').exists(): continue
        tasks = json.loads((out / 'tasks.json').read_text())
        axis = axes[1] if '__rate' in job or job == 'b5_cbp__s0' else axes[0]
        axis.plot([r['task'] for r in tasks], [r['online_error'] for r in tasks], label=job)
    for axis, title in zip(axes, ('Seed 0 arms', 'CBP rate checks')):
        axis.set(xlabel='Task', ylabel='Online error (lower is better)', title=title)
        axis.legend(fontsize=7); axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(STAGE_B / RUN_TAG / 'seed0_curves.png', dpi=150)
    plt.show()
    baseline = table[table.job == 'b5_cbp__s0']
    if not baseline.empty and baseline.iloc[0].status == 'COMPLETED':
        baseline = baseline.iloc[0]
        better = table[table.job.str.contains('__rate') & (table.status == 'COMPLETED') &
                       (table.mean_last50 < baseline.mean_last50)]
        if not better.empty:
            print('CBP RATE CHECK: these rates beat 1e-3 on seed 0; rate remains fixed at 1e-3.')
            print(better[['job', 'mean_last50']].to_string(index=False))
    print('Seed-0 preview only; original source commits are retained per trajectory. '
          'The registered three-seed conclusions remain pending.')
'''


def build_seed0(commit, seed=0):
    if seed not in (0, 1, 2): raise ValueError('B5 seeds must be 0, 1 or 2')
    notebook = build(commit, 'P-B5', f'vna_v3_b5_seed{seed}_resume')
    cells = notebook['cells']
    jobs = [job for job in PRESETS['P-B5'][2] if job.endswith(f'__s{seed}')]
    config = ''.join(cells[0]['source'])
    config = config.replace(f"JOBS_CPU = {PRESETS['P-B5'][2]!r}", f'JOBS_CPU = {jobs!r}')
    cells[0] = cell('code', config)
    cells[1] = cell('markdown', '''# B5 full seed-0 preview
CPU only; four workers; 12 arms + four baseline CBP rate checks = 16 jobs.
Every job runs the full 100-task horizon. Main CBP rate is fixed at 1e-3.
Progress prints every ten minutes and after each job. Final output includes
scores, curves, rate checks, checkpoints and an archive. No three-seed claims.

Use a fresh session with Internet enabled; source ZIP and CIFAR input are unnecessary.
Attach the old P-B5 Output/archive, this notebook's earlier Output/archive, or a standalone
resume.pkl as Input. Every selected source is printed before training. Completed jobs
are validated and skipped. Unfinished jobs continue using their original source commit;
their existing model, optimizer, RNG and hook state are retained. A full checkpoint
without summary.json is replayed without training to export its final results.
Legacy GradMax from before the numerical repair is restarted using the fixed code.
The table records the actual source commit/tag; trajectories are not relabelled.
If no reusable checkpoint/result is found, the notebook stops instead of silently starting all jobs over.
''')
    runtime = PROGRESS + ''.join(cells[4]['source'])
    runtime = runtime.replace('print_heartbeat(RUN_ROOT / "outputs", SESSION_START, VNA_DEADLINE, lane_state)',
        'print_heartbeat(RUN_ROOT / "outputs", SESSION_START, VNA_DEADLINE, lane_state)\n        seed0_progress()')
    cells[4] = cell('code', runtime)
    runner = ''.join(cells[-1]['source'])
    runner = runner.replace('from concurrent.futures import ThreadPoolExecutor',
        'from concurrent.futures import ThreadPoolExecutor, as_completed')
    runner = runner.replace(
        'futures=[gpu_pool.submit(run_gpu,0,JOBS_GPU0),gpu_pool.submit(run_gpu,1,JOBS_GPU1)]',
        'futures=[gpu_pool.submit(run_gpu,gpu,jobs) for gpu,jobs in enumerate((JOBS_GPU0,JOBS_GPU1)) if jobs]')
    runner = runner.replace('try:\n    v3jobs =', '''seed0_sources = select_sources(Path('/kaggle/input'), STAGE_B, JOBS_CPU, RUN_TAG, REPO_COMMIT)
seed0_checkouts = checkout_sources(REPO, seed0_sources)

def run_cpu(job):
    try:
        if pause_due(): return
        out = seed0_out(job)
        source = seed0_sources.get(job, {})
        commit = source.get('commit', REPO_COMMIT)
        if (out / 'summary.json').exists():
            summary = json.loads((out / 'summary.json').read_text())
            if summary.get('status') == 'COMPLETED':
                validate_result(job, summary, json.loads((out / 'protocol.json').read_text()),
                    json.loads((out / 'tasks.json').read_text()), commit, SOURCE_DIGEST)
                print(f'SKIP_COMPLETED {job}: {out}', flush=True)
                return
        cmd = command(job)
        if cmd is None: return
        cmd[cmd.index('--out')+1] = str(out)
        checkout = seed0_checkouts.get(commit, REPO)
        if commit != REPO_COMMIT and '--schedule' in cmd:
            cmd[cmd.index('--schedule')+1] = str(out.parent / '_schedules/regression.json')
        if (out / 'resume.pkl').exists():
            print(f'LOAD_CHECKPOINT {job}: {out / "resume.pkl"} commit={commit}', flush=True)
            cmd = cmd[:2] + ['-c', RESUME_ENTRY] + cmd[4:]
        env = {**ENV, 'CUDA_VISIBLE_DEVICES': ''}
        env['PYTHONPATH'] = env.get('PYTHONPATH', '').replace(str(REPO), str(checkout))
        from experiments.vna_resume import launch_logged
        return launch_logged(cmd, cwd=checkout, env=env,
            log_path=RUN_ROOT/f'{RUN_TAG}_{job}.log', lane=job, job=job,
            lanes=lane_state, paused=session_paused, deadline=VNA_DEADLINE)
    except Exception as error:
        seed0_failures[job] = str(error)
        raise

seed0_progress()
try:
    v3jobs =''')
    runner = runner.replace('for future in futures: future.result()', '''for future in as_completed(futures):
                try: future.result()
                except Exception as error: print('JOB FAILED:', error, flush=True)
                seed0_progress()
                archive_outputs()''')
    runner = runner.replace('report(collect([ROOT.parent],RUN_TAG))', 'seed0_results()\narchive_outputs()')
    runner += "\nif seed0_failures: raise RuntimeError(f'Seed-0 jobs failed: {seed0_failures}; see per-job logs')\n"
    if seed:
        runner = runner.replace(
            "seed0_sources = select_sources(Path('/kaggle/input'), STAGE_B, JOBS_CPU, RUN_TAG, REPO_COMMIT)",
            f"seed0_sources = select_sources(Path('/kaggle/input'), STAGE_B, JOBS_CPU, RUN_TAG, REPO_COMMIT, seed={seed})\n" + '''fresh_commit = prepare_seed_schedule(Path('/kaggle/input'), STAGE_B, RUN_TAG, REPO_COMMIT)
for job in JOBS_CPU:
    if job not in seed0_sources:
        seed0_sources[job] = dict(out=ROOT/job, commit=fresh_commit, step=0, completed=False)
from experiments.stage_b5 import write_json
write_json(ROOT/'resume_sources.json', {job: {**source, 'out': str(source['out'])}
           for job, source in seed0_sources.items()})''')
        title = ''.join(cells[1]['source'])
        title = title.replace('12 arms + four baseline CBP rate checks = 16 jobs.',
                              '12 registered arms; the four CBP rate checks remain seed-0 only.')
        title = title.replace('If no reusable checkpoint/result is found, the notebook stops instead of silently starting all jobs over.',
            'Attach the same seed-0 Output/archive to seeds 1 and 2. Both reuse its original\n'
            'authority schedule without recalibration, using compatible repaired code for new jobs. Jobs without checkpoints\n'
            'start fresh; checkpoints from other seeds are not used as training state.')
        cells[1] = cell('markdown', title)
    cells[-1] = cell('code', runner)
    if seed:
        for item in cells:
            text = ''.join(item['source'])
            for old, new in (('seed0', f'seed{seed}'), ('seed-0', f'seed-{seed}'),
                             ('Seed-0', f'Seed-{seed}'), ('SEED 0', f'SEED {seed}'),
                             ('Seed 0 arms', f'Seed {seed} arms'), ('__s0', f'__s{seed}')):
                text = text.replace(old, new)
            # The shared loader file keeps its existing import path.
            text = text.replace(f'experiments.b5_seed{seed}_resume', 'experiments.b5_seed0_resume')
            item['source'] = text.splitlines(keepends=True)
        title = ''.join(cells[1]['source'])
        title = title.replace(f'the four CBP rate checks remain seed-{seed} only.',
                              'the four CBP rate checks remain seed-0 only.')
        title = title.replace(f'the same repaired seed-{seed} Output/archive',
                              'the same repaired seed-0 Output/archive')
        title = title.replace(f'the same seed-{seed} Output/archive', 'the same seed-0 Output/archive')
        cells[1] = cell('markdown', title)
        progress = ''.join(cells[4]['source']).replace("'CBP rate checks'", "'CBP reference (rate=1e-3)'")
        cells[4] = cell('code', progress)
    return notebook


def build_group(commit, group):
    arms = B5_GROUPS[group]
    notebook = build_seed0(commit, 1)
    cells = notebook['cells']
    jobs = [f'b5_{arm}__s{seed}' for seed in (1, 2) for arm in arms]
    config = ''.join(cells[0]['source'])
    lines = config.splitlines(keepends=True)
    lines = [f'JOBS_CPU = {jobs!r}\n' if line.startswith('JOBS_CPU = ') else
             f"RUN_TAG = 'vna_v3_b5_group{group}'\n" if line.startswith('RUN_TAG = ') else line
             for line in lines]
    cells[0] = cell('code', ''.join(lines))
    cells[1] = cell('markdown', f'''# B5 group {group}: seeds 1 and 2
Four arms: {', '.join(arms)}. Each arm runs both seeds: eight full jobs,
100 tasks/job, with at most four CPU workers. CBP rate remains 1e-3;
the four alternative rate checks remain in the seed-0 notebook.

Run groups A/B/C in three separate CPU sessions. Attach the same
seed-0 Output/archive with real B5 preflight to all three, plus any earlier
B5 Outputs/archives to resume. Each group only runs its own arms and seeds.
Completed results are validated and skipped; unfinished checkpoints retain
their original code commit. Missing jobs start fresh using the shared authority
schedule under compatible repaired code without another preflight. RESUME_APPLIED confirms restoration.
Pre-repair GradMax is restarted with the numerical repair. Save each group's
Output/archive and attach it on the next session. Source ZIP and CIFAR are unnecessary.
''')
    runtime = ''.join(cells[4]['source'])
    runtime = runtime.replace(
        "axis = axes[1] if '__rate' in job or job == 'b5_cbp__s1' else axes[0]",
        "axis = axes[0] if job.endswith('__s1') else axes[1]")
    runtime = runtime.replace("('Seed 1 arms', 'CBP reference (rate=1e-3)')", "('Seed 1', 'Seed 2')")
    cells[4] = cell('code', runtime)
    runner = ''.join(cells[-1]['source'])
    runner = runner.replace(
        "seed1_sources = select_sources(Path('/kaggle/input'), STAGE_B, JOBS_CPU, RUN_TAG, REPO_COMMIT, seed=1)",
        '''seed1_sources = {}
for active_seed in (1, 2):
    seed_jobs = [job for job in JOBS_CPU if job.endswith(f'__s{active_seed}')]
    seed1_sources.update(select_sources(Path('/kaggle/input'), STAGE_B, seed_jobs,
        RUN_TAG, REPO_COMMIT, seed=active_seed))''')
    cells[-1] = cell('code', runner)
    for item in cells:
        text = ''.join(item['source']).replace('seed1_', f'group{group}_')
        text = text.replace('SEED 1:', f'GROUP {group}, SEEDS 1+2:')
        text = text.replace('Seed-1 preview', f'Group {group} preview (seeds 1+2)')
        text = text.replace('Seed-1 jobs failed', f'Group {group} jobs failed')
        item['source'] = text.splitlines(keepends=True)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--group', choices=tuple(B5_GROUPS))
    parser.add_argument('--output')
    args = parser.parse_args()
    suffix = f'group{args.group}' if args.group else f'seed{args.seed}'
    output = Path(args.output or f'notebooks/kaggle_vna_parallel_P-B5_{suffix}.ipynb')
    notebook = build_group(args.commit, args.group) if args.group else build_seed0(args.commit, args.seed)
    output.write_text(json.dumps(notebook, indent=1)+'\n')


if __name__ == '__main__': main()
