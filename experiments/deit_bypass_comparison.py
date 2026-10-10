"""Compare only completed original-space trajectories from identical forks."""
import json
from pathlib import Path
from experiments.shared_protocol import atomic_json_save
from experiments.run_deit_projection_aware import write_csv


def compare(output, fork_hash, protocol, horizon, *, statuses=None):
    output=Path(output);rows=[];results={}
    for method in ('e_driven_o_raw','deit_bypass'):
        if statuses is not None and statuses.get(method, {}).get('status') != 'completed':
            rows.append({'method':method,'available':False,
                'status':statuses.get(method, {}).get('status','not_requested'),
                'reason':'current arm did not complete; existing result file ignored'})
            continue
        path=output/'arms'/method/'result.json'
        if not path.exists():
            rows.append({'method':method,'available':False});continue
        result=json.loads(path.read_text());identity=result['run_identity']
        completed=result.get('completed_epochs')
        if completed is None:
            completed=result['history'][-1]['post_fork_epoch']
        if (result['theta_best_hash']!=fork_hash or result['protocol']!=protocol
                or identity['post_fork_epochs']!=horizon or completed!=horizon):
            raise ValueError('E→O/Bypass comparison requires identical fork/protocol and completed shared horizon')
        eligible=result.get('accuracy_comparison_eligible',True)
        results[method]=result
        rows.append({'method':method,'available':True,'accuracy_comparison_eligible':eligible,
            'report_best_accuracy':result['report_best_accuracy'], 'report_best_loss':result['report_best_loss'],
            'report_best_epoch':result['report_best_epoch'],'delta_vs_historical_best':result['delta_vs_historical_best'],
            'scientific_escape':result['scientific_escape'],'rollback_count':len(result.get('rollback_events',[])),
            'bypass_completed':result.get('bypass_completed'), 'report_best_scope':result.get('report_best_scope'),
            'fork_hash':fork_hash,'seed':protocol['seed']})
    a=results.get('e_driven_o_raw');b=results.get('deit_bypass')
    eligible=bool(a and b and b['accuracy_comparison_eligible'])
    result={'rows':rows,'e2o_minus_bypass_best_accuracy':a['report_best_accuracy']-b['report_best_accuracy'] if eligible else None,
        'comparison_eligible':eligible,
        'protocol_difference':'E→O uses patience10 anchor rollback; Bypass uses opt1/opt2/contraction/train3 without rollback',
        'budget_note':f'Both have{horizon} training epochs; E→O proposal/CG overhead and expanded Bypass compute differ'}
    atomic_json_save(result,output/'bypass_comparison.json');write_csv(rows,output/'bypass_comparison.csv')
    return result
