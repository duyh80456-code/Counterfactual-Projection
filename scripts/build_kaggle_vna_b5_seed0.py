"""Build the full B5 seed-0 preview using the shared resumable pilot runner."""
import argparse
import json
from pathlib import Path

from scripts.build_kaggle_vna_parallel import build, PRESETS
from scripts.build_kaggle_vna_signal_curve_notebook import cell


PROGRESS = r'''
from scripts.collect_parallel import validate_result
seed0_failures = {}

def seed0_progress():
    directory = STAGE_B / RUN_TAG
    rows = []
    for job in JOBS_CPU:
        out = directory / job
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
            validate_result(job, summary, protocol, tasks, REPO_COMMIT, SOURCE_DIGEST)
        state = ('FAILED' if job in seed0_failures else 'COMPLETED' if done else
                 'PAUSED' if session_paused.is_set() and status else
                 'RUNNING' if status else 'PENDING')
        rows.append(dict(job=job, status=state, tasks=len(tasks), total=expected,
            step=status.get('step'), online_error=tasks[-1].get('online_error') if tasks else None,
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
        out = STAGE_B / RUN_TAG / job
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
    baseline = table[table.job == 'b5_cbp__s0'].iloc[0]
    if baseline.status == 'COMPLETED':
        better = table[table.job.str.contains('__rate') & (table.status == 'COMPLETED') &
                       (table.mean_last50 < baseline.mean_last50)]
        if not better.empty:
            print('CBP RATE CHECK: these rates beat 1e-3 on seed 0; rate remains fixed at 1e-3.')
            print(better[['job', 'mean_last50']].to_string(index=False))
    print('Seed-0 preview only; the registered three-seed conclusions remain pending.')
'''


def build_seed0(commit):
    notebook = build(commit, 'P-B5', 'vna_v3_b5_seed0')
    cells = notebook['cells']
    jobs = [job for job in PRESETS['P-B5'][2] if job.endswith('__s0')]
    config = ''.join(cells[0]['source'])
    config = config.replace(f"JOBS_CPU = {PRESETS['P-B5'][2]!r}", f'JOBS_CPU = {jobs!r}')
    cells[0] = cell('code', config)
    cells[1] = cell('markdown', '''# B5 full seed-0 preview
CPU only; four workers; 12 arms + four baseline CBP rate checks = 16 jobs.
Every job runs the full 100-task horizon. Main CBP rate is fixed at 1e-3.
Progress prints every ten minutes and after each job. Final output includes
scores, curves, rate checks, checkpoints and an archive. No three-seed claims.

Use a fresh session with Internet enabled; source ZIP and CIFAR input are unnecessary.
To resume this notebook, attach its previous archive and retain the same commit and RUN_TAG.
Older P-B5 archives use another commit/tag and are not reused as revised training results.
''')
    runtime = PROGRESS + ''.join(cells[4]['source'])
    runtime = runtime.replace('print_heartbeat(RUN_ROOT / "outputs", SESSION_START, VNA_DEADLINE, lane_state)',
        'print_heartbeat(RUN_ROOT / "outputs", SESSION_START, VNA_DEADLINE, lane_state)\n        seed0_progress()')
    cells[4] = cell('code', runtime)
    runner = ''.join(cells[-1]['source'])
    runner = runner.replace('from concurrent.futures import ThreadPoolExecutor',
        'from concurrent.futures import ThreadPoolExecutor, as_completed')
    runner = runner.replace('try:\n    v3jobs =', '''native_run_cpu = run_cpu
def run_cpu(job):
    try: return native_run_cpu(job)
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
    cells[-1] = cell('code', runner)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', default='notebooks/kaggle_vna_parallel_P-B5_seed0.ipynb')
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(build_seed0(args.commit), indent=1)+'\n')


if __name__ == '__main__': main()
