"""Build the common parallel pilot notebook and its six first-cell variants."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.build_kaggle_pilot_common import notebook
from scripts.build_kaggle_vna_signal_curve_notebook import cell


PRESETS = {
 'P1': (['icarl_vna__inc10__s1993'], ['icarl_ngd__inc10__s1993'], ['b5_bp__s0','b5_cbp__s0','b5_cbp_gir_vna__s0']),
 'P2': (['icarl_vna_random__inc10__s1993'], ['icarl_vna_euclid__inc10__s1993'], ['b5_bp__s1','b5_cbp__s1','b5_cbp_gir_vna__s1']),
 'P3': (['icarl__inc10__s1994'], ['icarl_vna__inc10__s1994'], ['b5_bp__s2','b5_cbp__s2','b5_cbp_gir_vna__s2']),
 'P4': (['icarl__inc10__s1995'], ['icarl_vna__inc10__s1995'], ['b5_cbp_gir__s0','b5_cbp_gir__s1','b5_cbp_gir__s2']),
 'P5': (['icarl__inc5__s1993'], ['icarl_vna__inc5__s1993'], ['b5_cbp_gir_randdir__s0','b5_cbp_gir_randdir__s1','b5_cbp_gir_randdir__s2']),
 'P6': (['icarl__b50inc10__s1993'], ['icarl_vna__b50inc10__s1993'], ['b5_bp_vna__s0','b5_bp_vna__s1','b5_bp_vna__s2','b5_bp_ngd__s0','b5_bp_ngd__s1','b5_bp_ngd__s2']),
}


def config(commit, jobs, run_tag):
    gpu0, gpu1, cpu = jobs
    return f'''JOBS_GPU0 = {gpu0!r}
JOBS_GPU1 = {gpu1!r}
JOBS_CPU = {cpu!r}
RUN_TAG = {run_tag!r}

import time
SESSION_HOURS = 11.5
SESSION_START = time.time()
VNA_DEADLINE = SESSION_START + SESSION_HOURS*3600
REPO_COMMIT = {commit!r}
ALLOW_OVER_BUDGET = True
DISK_CAP_BYTES = 5_000_000_000
'''


RUNNER = r'''import re
from concurrent.futures import ThreadPoolExecutor
ROOT = STAGE_B / RUN_TAG
ROOT.mkdir(parents=True, exist_ok=True)
ENV.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

def completed(job):
    path = ROOT/job/"summary.json"
    if not path.exists(): return False
    summary = json.loads(path.read_text())
    if summary.get("status") != "COMPLETED": return False
    protocol_path = ROOT/job/"protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        commit = protocol.get("code_commit") or protocol.get("identity",{}).get("code_commit")
        if commit and commit != REPO_COMMIT: raise ValueError("completed commit mismatch: "+job)
    return True

def cil_command(job):
    match = re.fullmatch(r"(icarl(?:_vna(?:_random|_euclid)?|_ngd)?)__(inc5|inc10|b50inc10)__s(\d+)", job)
    if not match: raise ValueError("unknown CIL job "+job)
    label, protocol, seed = match.groups()
    arm = "icarl_ngd" if label == "icarl_ngd" else "icarl_vna" if label.startswith("icarl_vna") else "icarl"
    selection = "random_frob" if label.endswith("_random") else "tiny_V_euclid" if label.endswith("_euclid") else "tiny_V"
    init_cls, increment = (50,10) if protocol == "b50inc10" else (10,5 if protocol == "inc5" else 10)
    cmd = [sys.executable,"-u","-m","experiments.pilot_b3","--arm",arm,
           "--selection",selection,"--seed",seed,"--init-cls",str(init_cls),
           "--increment",str(increment),"--out",str(ROOT/job),"--device","cuda:0",
           "--workers","2","--data-dir",str(DATA_ROOT),"--no-measure"]
    if arm != "icarl": cmd += ["--auto-schedule"]
    return cmd

def b5_command(job):
    match = re.fullmatch(r"b5_(.+)__s([012])", job)
    if not match: raise ValueError("unknown B5 job "+job)
    label, seed = match.groups()
    arms = {"bp":"bp", "cbp":"cbp", "cbp_gir":"cbp_gir",
            "cbp_gir_randdir":"cbp_gir_randdir", "cbp_gir_vna":"cbp_gir_vna_l1",
            "bp_vna":"bp_vna_l1", "bp_ngd":"bp_ngd_l1"}
    return [sys.executable,"-u","-m","experiments.pilot_b5","--arm",arms[label],
            "--seed",seed,"--out",str(ROOT/job),"--flat-output","--resume"]

def run_gpu(gpu, jobs):
    for job in jobs:
        if pause_due(): break
        if completed(job): continue
        if not launch(cil_command(job), gpu, job, RUN_ROOT/f"{RUN_TAG}_{job}.log"): break
        archive_outputs()

def run_cpu(job):
    if pause_due() or completed(job): return
    from experiments.vna_resume import launch_logged
    return launch_logged(b5_command(job), cwd=REPO,
        env={**ENV,"CUDA_VISIBLE_DEVICES":""}, log_path=RUN_ROOT/f"{RUN_TAG}_{job}.log",
        lane="CPU", job=job, lanes=lane_state, paused=session_paused, deadline=VNA_DEADLINE)

def reuse_existing_baseline():
    target = ROOT/"icarl__inc10__s1993"
    if target.exists(): return
    for path in Path("/kaggle/input").rglob("summary.json"):
        try: summary = json.loads(path.read_text())
        except (OSError,ValueError): continue
        if (summary.get("arm") == "icarl" and summary.get("status") == "COMPLETED"
                and summary.get("seed",1993) == 1993 and summary.get("tasks_run") == 10):
            target.mkdir(parents=True, exist_ok=True)
            for name in ("summary.json","tasks.json","protocol.json","reproduction_gate.json","vna_log.csv"):
                source = path.parent/name
                if source.exists(): shutil.copy2(source,target/name)
            print("Reused attached inc10 seed1993 baseline:",path)
            return

try:
    reuse_existing_baseline()
    with ThreadPoolExecutor(max_workers=2) as gpu_pool, ThreadPoolExecutor(max_workers=4) as cpu_pool:
        futures = [gpu_pool.submit(run_gpu,0,JOBS_GPU0), gpu_pool.submit(run_gpu,1,JOBS_GPU1)]
        futures += [cpu_pool.submit(run_cpu,job) for job in JOBS_CPU]
        for future in futures: future.result()
finally:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs()

rows=[]
for path in sorted(ROOT.glob("*/summary.json")):
    summary=json.loads(path.read_text()); job=path.parent.name
    if summary.get("status") != "COMPLETED": continue
    tasks_path=path.parent/"tasks.json"
    tasks=json.loads(tasks_path.read_text()) if tasks_path.exists() else summary.get("tasks",[])
    if job.startswith("icarl") and tasks:
        last=tasks[-1]
        rows.append(dict(job=job, cnn_average=last.get("cnn_average_accuracy"),
            nme_average=last.get("nme_average_accuracy"), cnn_last=last.get("cnn_top1"),
            nme_last=last.get("nme_top1"), cnn_forgetting=last.get("cnn_forgetting"),
            nme_forgetting=last.get("nme_forgetting")))
    elif job.startswith("b5_"):
        rows.append(dict(job=job, mean_last50=summary.get("mean_last50"), variants=len(summary.get("variants",[]))))
import pandas as pd
display(pd.DataFrame(rows))
if session_paused.is_set(): print("PAUSED: attach this output and rerun to resume")
if archive_errors: raise RuntimeError(archive_errors)
'''


def build(commit, preset=None, run_tag='vna_parallel_20261017'):
    jobs = PRESETS[preset] if preset else ([],[],[])
    result = notebook(commit)
    cells = result['cells']
    cells[0] = cell('markdown', '# Parallel VNA pilot\nSix-account CIL/B5 run; full jobs start directly after one shared pytest pass.')
    cells[1] = cell('code', config(commit, jobs, run_tag))
    cells[-1] = cell('code', RUNNER)
    result['cells'] = [cells[1], cells[0], *cells[2:]]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--run-tag',default='vna_parallel_20261017')
    parser.add_argument('--output',default=str(ROOT/'notebooks/kaggle_vna_parallel.ipynb'))
    parser.add_argument('--variants',action='store_true')
    args=parser.parse_args()
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(build(args.commit,None,args.run_tag),indent=1)+'\n')
    if args.variants:
        for name in PRESETS:
            path=output.with_name(output.stem+'_'+name+output.suffix)
            path.write_text(json.dumps(build(args.commit,name,args.run_tag),indent=1)+'\n')


if __name__=='__main__': main()
