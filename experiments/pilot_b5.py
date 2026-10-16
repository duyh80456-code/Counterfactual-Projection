"""Performance-only paired LoP regression pilot; no Stage-B measurement pass."""
import argparse
import json
from pathlib import Path
import pickle
import random
import time
import numpy as np
import torch
from experiments import stage_b5 as native
from experiments.vna_resume import Progress, Status, rng_state
from adapters.lop_hooks import LearnHook, learner_state
from adapters.pilot_lop import LoPPilot
from diagnostics.vna_align import MeasurementBuffer
from experiments.pilot_gir_vna import PilotEngine

ARMS = ('bp', 'cbp', 'cbp_gir', 'cbp_gir_randdir', 'bp_vna_l0.1', 'bp_vna_l1',
        'bp_ngd_l0.1', 'bp_ngd_l1', 'cbp_gir_vna_l1')


def run(args):
    native.upstream_imports()
    if args.fake_data: torch.set_num_threads(1)
    base = 'cbp' if args.arm.startswith('cbp') else 'bp'
    if base == 'cbp' and args.replacement_rate is None:
        from copy import copy
        results = []
        for rate in native.RATES:
            child = copy(args); child.replacement_rate = rate
            if args.flat_output: child.out = str(Path(args.out)/'rates'/f'rr_{rate:g}')
            results.append(run(child))
        if args.flat_output:
            root = Path(args.out).resolve(); root.mkdir(parents=True, exist_ok=True)
            aggregate = dict(status='COMPLETED', arm=args.arm, seed=args.seed,
                             replacement_rate='all_registered', variants=results,
                             mean_last50=None, seconds=sum(r['seconds'] for r in results))
            native.write_json(root/'summary.json', aggregate)
            native.write_json(root/'tasks.json', {str(r['replacement_rate']):r['tasks'] for r in results})
            native.write_json(root/'protocol.json', dict(arm=args.arm, seed=args.seed,
                replacement_rates=native.RATES, code_commit=results[0]['identity']['code_commit']))
            combined = [{**event, 'replacement_rate': result['replacement_rate']}
                        for result in results for event in result['events']
                        if event.get('kind') == 'correction']
            if combined: native.write_csv(root/'vna_log.csv', combined)
            elif not (root/'vna_log.csv').exists():
                (root/'vna_log.csv').write_text('task,step,K,rho,seconds,selection,skipped,replacement_rate\n')
            return aggregate
        return results
    params, provenance = native.recipe('regression', base, args.replacement_rate)
    if args.smoke: params.update(num_data_points=1000, flip_after=500)
    params.update(run_idx=0, use_gpu=0)
    suffix = '' if base == 'bp' else f'_rr{args.replacement_rate:g}'
    out = (Path(args.out).resolve() if args.flat_output else
           Path(args.out).resolve()/f'{args.arm}{suffix}'/f'seed{args.seed}')
    out.mkdir(parents=True, exist_ok=True)
    params['data_file'] = str(out/'upstream_in_progress.tmp')
    params['env_file'] = str(out/'data/regression.pkl')
    identity = dict(arm=args.arm, seed=args.seed, params=params, fake_data=args.fake_data, smoke=args.smoke,
                    gir_pool=args.gir_pool, gir_refresh_steps=args.gir_refresh_steps, rho_min=args.rho_min,
                    code_commit=__import__('experiments.vna_provenance', fromlist=['code_commit']).code_commit())
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()) != identity:
        raise ValueError('pilot resume identity mismatch')
    native.write_json(out/'manifest.json', identity)
    native.write_json(out/'protocol.json', identity)
    if args.resume and (out/'summary.json').exists():
        previous = json.loads((out/'summary.json').read_text())
        if previous.get('status') == 'COMPLETED': return previous
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    native.prepare_data('regression', params, out, fake=args.fake_data, smoke=args.smoke)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    coefficient = float(args.arm.rsplit('_l', 1)[1]) if '_l' in args.arm else 0.
    engine = PilotEngine(seed=args.seed, gir='gir' in args.arm, random='randdir' in args.arm,
                         coefficient=coefficient, ngd='ngd' in args.arm,
                         pool=args.gir_pool, refresh=args.gir_refresh_steps, rho_min=args.rho_min)
    intervention = LoPPilot(engine)
    checkpoint_path = out/'resume.pkl'
    resumed = None
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open('rb') as f: resumed = pickle.load(f)
    def checkpoint(state):
        native.atomic_pickle(checkpoint_path, state)
        return checkpoint_path.stat().st_size
    status = Status(out, f'pilot B5 {args.arm} seed={args.seed}')
    hook = LearnHook('regression', base, params, seed=args.seed, checkpoint=checkpoint,
                     resume=resumed, intervention=intervention, progress=Progress(out, identity, name='deadline.pt'), status=status)
    hook.task_log_only = True
    hook.buffer = MeasurementBuffer(128)
    started = time.monotonic()
    try:
        from lop.slowly_changing_regression import expr as native_expr
        from tqdm import tqdm
        original_progress = native_expr.tqdm
        try:
            native_expr.tqdm = lambda iterable: tqdm(iterable, disable=True)
            with intervention.install(): logs = native.run_native('regression', params, out, hook)
        finally: native_expr.tqdm = original_progress
        hook._checkpoint(hook.last_learner, hook.last_frame)
        native.atomic_pickle(out/'final_state.pkl', dict(learner=learner_state(hook.last_learner), rng=rng_state()))
        native.atomic_pickle(out/'upstream.pkl', logs)
        tasks = list(hook.results.tasks())
        seconds = hook.elapsed_base+time.monotonic()-started
        errors = [r['online_error'] for r in tasks]
        summary = dict(identity=identity, arm=args.arm, seed=args.seed, replacement_rate=args.replacement_rate,
            status='COMPLETED', tasks=tasks, mean_last50=float(np.mean(errors[-50:])),
            mean_all=float(np.mean(errors)), seconds=seconds, steps=hook.step, seconds_per_step=seconds/hook.step,
            events=engine.events, activities=[dict(task=r['task'], sites=[{k:v for k,v in a.items() if k!='dormant_indices'}
                     for a in r['sites']]) for r in engine.activities],
            estimated_full_seconds=seconds/hook.step*1_000_000)
        summary.update({k:v for k,v in provenance.items() if k!='replacement_rate'})
        correction_seconds = sum(r['seconds'] for r in engine.events if r['kind']=='correction')
        gir_seconds = sum(r['seconds'] for r in engine.events if r['kind']=='gir_refresh')
        summary.update(correction_seconds=correction_seconds, gir_refresh_seconds=gir_seconds,
            train_seconds_per_step=max(0.,seconds-correction_seconds-gir_seconds)/hook.step,
            gir_random_fill=sum(r.get('gir_random_fill',0) for r in engine.events),
            rho_gate_skips=sum(r.get('rho_gate_skip',False) for r in engine.events))
        native.write_json(out/'summary.json', summary)
        native.write_json(out/'tasks.json', tasks)
        native.write_csv(out/'online.csv', tasks)
        corrections = [row for row in engine.events if row.get('kind') == 'correction']
        if corrections: native.write_csv(out/'vna_log.csv', corrections)
        elif not (out/'vna_log.csv').exists():
            (out/'vna_log.csv').write_text('task,step,K,rho,seconds,selection,skipped\n')
        print(json.dumps({k:v for k,v in summary.items() if k not in ('events','activities','identity','tasks')}), flush=True)
        return summary
    finally: status.close()


def comparison(root):
    """Only compare like replacement rates; pair every delta by seed."""
    rows = [json.loads(p.read_text()) for p in Path(root).glob('*/seed*/summary.json')]
    table = []
    for arm, rate in sorted({(r['arm'], r['replacement_rate']) for r in rows}, key=str):
        selected = [r for r in rows if r['arm']==arm and r['replacement_rate']==rate]
        values = [r['mean_last50'] for r in selected]
        row = dict(arm=arm, replacement_rate=rate, seeds=len(values), mean=np.mean(values),
                   std=np.std(values, ddof=1) if len(values)>1 else None)
        for baseline in ('bp','cbp'):
            controls = {r['seed']:r for r in rows if r['arm']==baseline and
                        (baseline=='bp' or r['replacement_rate']==rate)}
            # BP rows have no rr: show one CBP comparison for each registered rate.
            rates = native.RATES if baseline=='cbp' and rate is None else (rate,)
            for control_rate in rates:
                if baseline=='cbp': controls = {r['seed']:r for r in rows if r['arm']=='cbp' and r['replacement_rate']==control_rate}
                diffs = [r['mean_last50']-controls[r['seed']]['mean_last50'] for r in selected if r['seed'] in controls]
                key = baseline+(f'_rr{control_rate:g}' if baseline=='cbp' else '')
                row['delta_'+key] = float(np.mean(diffs)) if diffs else None
                row['delta_std_'+key] = float(np.std(diffs, ddof=1)) if len(diffs)>1 else None
        table.append(row)
    return table


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=(0,1,2), default=0)
    p.add_argument('--replacement-rate', type=float, choices=native.RATES)
    p.add_argument('--out', default='outputs/pilot_b5')
    p.add_argument('--flat-output', action='store_true')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--fake-data', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--gir-pool', type=int, default=8)
    p.add_argument('--gir-refresh-steps', type=int, default=1000)
    p.add_argument('--rho-min', type=float, default=0.)
    return p

if __name__ == '__main__': run(parser().parse_args())
