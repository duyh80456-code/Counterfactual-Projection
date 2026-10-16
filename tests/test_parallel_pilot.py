"""Parallel pilot CLI, notebook and collection contracts."""
import ast
import json
from pathlib import Path

from experiments.pilot_b3 import parser, remeasure, remeasure_parser
from scripts.build_kaggle_vna_parallel import build, PRESETS
from scripts.collect_parallel import collect


def test_icarl_parallel_arguments_and_missing_remeasure_states(tmp_path):
    args = parser().parse_args(['--arm','icarl_vna','--selection','tiny_V_euclid',
        '--seed','1995','--init-cls','50','--increment','10','--auto-schedule'])
    assert (args.seed,args.init_cls,args.increment,args.selection,args.auto_schedule) == (1995,50,10,'tiny_V_euclid',True)
    request = remeasure_parser().parse_args(['--from',str(tmp_path),'--device','cpu'])
    try: remeasure(request)
    except FileNotFoundError as error: assert 'measurement checkpoints' in str(error)
    else: raise AssertionError('remeasure must reject a run without point checkpoints')


def test_parallel_notebook_variants_only_change_first_cell():
    commit = 'a'*40
    notebooks = [build(commit, name, 'test-run') for name in PRESETS]
    for notebook in notebooks:
        for item in notebook['cells']:
            if item['cell_type'] == 'code': ast.parse(''.join(item['source']))
        first = ''.join(notebook['cells'][0]['source'])
        assert all(key in first for key in ('JOBS_GPU0','JOBS_GPU1','JOBS_CPU','RUN_TAG'))
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        assert commit in code and 'max_workers=4' in code and 'ALLOW_OVER_BUDGET = True' in code
    assert all(notebook['cells'][1:] == notebooks[0]['cells'][1:] for notebook in notebooks[1:])


def test_collector_reads_run_tag_and_standard_artifacts(tmp_path):
    job = tmp_path/'outputs'/'tag-x'/'icarl__inc10__s1994'; job.mkdir(parents=True)
    (job/'summary.json').write_text(json.dumps({'status':'COMPLETED','seed':1994,'tasks':[]}))
    (job/'tasks.json').write_text('[]')
    (job/'protocol.json').write_text(json.dumps({'seed':1994,'init_cls':10,'increment':10}))
    (job/'vna_log.csv').write_text('task,step,K,rho,seconds\n1,1,100,.2,3\n')
    result = collect([tmp_path], 'tag-x')
    assert set(result) == {'icarl__inc10__s1994'}
    assert result['icarl__inc10__s1994']['vna'][0]['K'] == '100'
