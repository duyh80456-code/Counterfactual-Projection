"""Parallel pilot CLI, notebook and collection contracts."""
import ast
import json
from pathlib import Path
import copy
import pytest

from experiments.pilot_b3 import parser, remeasure, remeasure_parser
from scripts.build_kaggle_vna_parallel import build, PRESETS
from scripts.build_kaggle_vna_parallel import RUNNER
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
    summary, protocol, tasks = valid_cil()
    (job/'summary.json').write_text(json.dumps(summary))
    (job/'tasks.json').write_text(json.dumps(tasks))
    (job/'protocol.json').write_text(json.dumps(protocol))
    (job/'vna_log.csv').write_text('task,step,K,rho,seconds\n1,1,100,.2,3\n')
    result = collect([tmp_path], 'tag-x')
    assert set(result) == {'icarl__inc10__s1994'}
    assert result['icarl__inc10__s1994']['vna'][0]['K'] == '100'


def valid_cil(seed=1994, commit='a'*40):
    recipe = dict(dataset='cifar100',init_cls=10,increment=10,seed=[seed],
                  memory_size=2000,convnet_type='resnet32')
    protocol = dict(seed=seed,init_cls=10,increment=10,arm='icarl',code_commit=commit,
        upstream_commit='b'*40,dataset_fingerprint={k:'c'*64 for k in ('train','test','meta')},
        fake_data=False,smoke=False,pilot=False,args=recipe,workers=2)
    tasks = [dict(task=i) for i in range(1,11)]
    summary = dict(protocol, status='COMPLETED',tasks=tasks,tasks_run=10,
        identity=dict(protocol,recipe=recipe,init_seed=0))
    return summary, protocol, tasks


def write_result(root, seed=1994, commit='a'*40, change=None):
    summary, protocol, tasks = valid_cil(seed,commit)
    if change: change(summary,protocol,tasks)
    directory = root/'outputs'/'tag'/f'icarl__inc10__s{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    for name,value in [('summary',summary),('protocol',protocol),('tasks',tasks)]:
        (directory/(name+'.json')).write_text(json.dumps(value))
    return directory


def test_collector_ignores_incomplete_and_nested_results(tmp_path):
    directory = write_result(tmp_path, change=lambda s,p,t:s.update(status='PAUSED'))
    nested = directory/'rates'/'rr_0.1'; nested.mkdir(parents=True)
    (nested/'summary.json').write_text('{"status":"COMPLETED"}')
    assert collect([tmp_path], 'tag') == {}


@pytest.mark.parametrize('change',[
    lambda s,p,t:p.update(code_commit='d'*40),
    lambda s,p,t:p.pop('dataset_fingerprint'),
    lambda s,p,t:p.update(smoke=True),
    lambda s,p,t:p.update(increment=5),
    lambda s,p,t:t.pop(),
    lambda s,p,t:p['args'].update(dataset='cifar10'),
])
def test_collector_rejects_invalid_completed_records(tmp_path, change):
    write_result(tmp_path,change=change)
    with pytest.raises(ValueError): collect([tmp_path], 'tag')


def test_collector_rejects_mixed_commits_protocols_and_duplicates(tmp_path):
    one,two = tmp_path/'one',tmp_path/'two'
    write_result(one)
    write_result(two,1995,'d'*40)
    with pytest.raises(ValueError,match='code_commit'): collect([one,two],'tag')
    write_result(two,1995,change=lambda s,p,t:p['args'].update(memory_size=500))
    with pytest.raises(ValueError,match='protocol'): collect([one,two],'tag')
    # Same job and protocol, but conflicting output metrics must not win by order.
    other = tmp_path/'duplicate'
    write_result(other,change=lambda s,p,t:s.update(seconds=10))
    with pytest.raises(ValueError,match='duplicate'): collect([one,other],'tag')


def test_baseline_requires_exact_recipe_and_dataset(tmp_path,monkeypatch):
    import experiments.parallel_validity as validity
    from experiments.stage_b3 import recipe,upstream_imports,UPSTREAM_PIN
    upstream_imports()
    summary,protocol,tasks=valid_cil(1993)
    args=recipe()
    protocol.update(args=args,upstream_commit=UPSTREAM_PIN)
    summary['identity'].update(recipe=args,upstream_commit=UPSTREAM_PIN)
    directory=tmp_path/'baseline'; directory.mkdir()
    def save():
        for name,value in [('summary',summary),('protocol',protocol),('tasks',tasks)]:
            (directory/(name+'.json')).write_text(json.dumps(value))
    save()
    monkeypatch.setattr(validity,'dataset_fingerprint',lambda root:protocol['dataset_fingerprint'])
    validity.validate_baseline(directory,'a'*40,tmp_path)
    protocol['args']['memory_size']=500; save()
    with pytest.raises(ValueError,match='recipe'): validity.validate_baseline(directory,'a'*40,tmp_path)


def test_b5_validator_rejects_partial_rates_and_protocol_mismatch():
    from experiments.parallel_validity import validate_record
    identity=dict(code_commit='a'*40,fake_data=False,smoke=False,
                  params=dict(num_data_points=100,flip_after=50))
    tasks=[dict(task=1),dict(task=2)]
    summary=dict(status='COMPLETED',arm='bp',seed=0,steps=100,
                 tasks=tasks,identity=identity)
    validate_record('b5_bp__s0',summary,identity,tasks)
    with pytest.raises(ValueError,match='identity mismatch'):
        validate_record('b5_bp__s0',summary,dict(identity,extra=True),tasks)
    summary.update(arm='cbp',replacement_rate=.001)
    aggregate=dict(status='COMPLETED',arm='cbp',seed=0,variants=[summary])
    protocol=dict(code_commit='a'*40,replacement_rates=[.001,.01])
    with pytest.raises(ValueError,match='registered rate'):
        validate_record('b5_cbp__s0',aggregate,protocol,{'.001':tasks})


def test_t4_preflight_uses_separate_output_and_calibration(tmp_path):
    import re
    namespace=dict(re=re,sys=__import__('sys'),ROOT=tmp_path,DATA_ROOT=tmp_path,
                   RUN_ROOT=tmp_path,RUN_TAG='test')
    tree=ast.parse(RUNNER)
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<runner>','exec'),namespace)
    launched=[]
    namespace['launch']=lambda command,*args:launched.append(command) or True
    namespace['real_t4_preflight'](0,['icarl_vna_euclid__inc10__s1993'])
    cmd=launched[0]
    assert '--smoke' in cmd and '--calibrate' in cmd and '--auto-schedule' not in cmd
    assert '_preflight' in cmd[cmd.index('--out')+1]
    guard=next(node for node in tree.body if isinstance(node,ast.Try))
    stop=next(i for i,node in enumerate(guard.body) if isinstance(node,ast.Raise))
    baseline=next(i for i,node in enumerate(guard.body) if isinstance(node,ast.Expr)
                  and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                  and node.value.func.id=='reuse_existing_baseline')
    assert stop < baseline
