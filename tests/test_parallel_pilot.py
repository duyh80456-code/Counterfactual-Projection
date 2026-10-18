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



def runner_functions(tmp_path, preset='M1'):
    import re
    namespace=dict(re=re,sys=__import__('sys'),ROOT=tmp_path,DATA_ROOT=tmp_path,
                   RUN_ROOT=tmp_path,RUN_TAG='test',Path=Path,json=json,
                   REPO_COMMIT='a'*40,SOURCE_DIGEST='c'*64,PRESET=preset)
    tree=ast.parse(RUNNER)
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<runner>','exec'),namespace)
    return namespace


def test_authority_preflight_produces_shared_schedule_and_no_hold(tmp_path):
    namespace=runner_functions(tmp_path)
    launched=[]
    namespace.update(archive_outputs=lambda: None, pause_due=lambda: False)
    def launch(command,*args):
        launched.append(command)
        path = Path(command[command.index('--schedule-output')+1])
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{}')
        return True
    namespace['launch']=launch
    namespace['prepare_schedule']('nobn')
    cmd=launched[0]
    assert '--preflight-only' in cmd and cmd[cmd.index('--arm')+1]=='icarl_full'
    assert 'experiments.main_cil' in cmd
    assert cmd[cmd.index('--schedule-source')+1]=='authority'
    cmd=namespace['command']('icarl_full__nobn__superclass__s0')
    assert Path(cmd[cmd.index('--schedule')+1]).name=='icarl_noBN.json'
    (tmp_path/'_schedules'/'icarl_noBN.json').unlink()
    namespace['PRESET']='M3'
    namespace['prepare_schedule']('nobn')
    assert launched[-1][launched[-1].index('--schedule-source')+1]=='self'


def test_v4_notebooks_build_and_expected_jobs():
    counts={'M1':6,'M2':6,'M3':6,'C1':4,'C2':24,'C3':12,'C4':12,'C5':1,'B5w':36}
    assert set(PRESETS)==set(counts)
    for name, jobs in PRESETS.items():
        assert sum(map(len,jobs))==counts[name]
        value=json.loads(json.dumps(build('PLACEHOLDER',name)))
        assert value['nbformat']==4 and isinstance(value['metadata'],dict)
        assert all(c['cell_type'] in ('code','markdown') and isinstance(c['source'],list) for c in value['cells'])
        config={}
        exec(''.join(value['cells'][0]['source']),config)
        assert config['REPO_COMMIT']=='PLACEHOLDER'
        assert (config['JOBS_GPU0'],config['JOBS_GPU1'],config['JOBS_CPU'])==jobs
        for item in value['cells']:
            if item['cell_type']=='code': ast.parse(''.join(item['source']))
        final=''.join(value['cells'][-1]['source'])
        assert all(token in final for token in ('report(records)','stability_plasticity','gir_fallback','warm_reset','rejected','skip_reason'))
    assert all('sweep' in job for lane in PRESETS['C1'] for job in lane)
    assert {j.split('__')[1] for lane in PRESETS['C2'] for j in lane}=={'bn','nobn'}
    namespace=runner_functions(Path('/tmp/v4-jobs'))
    for name in ('M1','M2','M3','C1','C2','C5','B5w'):
        namespace['PRESET']=name
        for lane in PRESETS[name]:
            for job in lane:
                cmd=namespace['command'](job)
                assert '--out' in cmd and '--device' in cmd


def test_collector_shared_schedule_and_paired_decisions():
    from scripts.collect_parallel import paired, decisions, validate_shared_schedules
    jobs={}
    for arm in ('b5_bp_vna','b5_bp_kself','b5_bp_vna_rand'):
        for seed in range(3):
            jobs[f'{arm}__s{seed}']=dict(summary=dict(seed=seed,mean_last50=1 if arm=='b5_bp_vna' else 2),
                protocol=dict(setting='regression',schedule=dict(K_base=50,mu_rel=.01)))
    row=paired(jobs,'b5_bp_vna','b5_bp_kself','mean_last50',True)
    assert row['wins']==3 and row['complete'] and row['sign_test_p_one_sided']==.125
    assert decisions(jobs)[0]['signal']
    validate_shared_schedules(jobs)
    jobs['b5_bp_kself__s0']['protocol']['schedule']=dict(K_base=100,mu_rel=.01)
    with pytest.raises(ValueError,match='shared authority'):
        validate_shared_schedules(jobs)


def test_v3_collector_rejects_synthetic_partial_or_mixed_source(tmp_path):
    from scripts.collect_parallel import validate_result
    protocol=dict(algorithm_version='v3',setting='regression',arm='b5_bp',seed=0,
        code_commit='a'*40,source_digest='c'*64,upstream_commit='b'*40,
        synthetic=False,preflight_only=False,cfg=dict(num_data_points=100,flip_after=50))
    tasks=[dict(task=1),dict(task=2)]
    summary=dict(status='COMPLETED',identity=protocol,tasks=tasks)
    validate_result('b5_bp__s0',summary,protocol,tasks,'a'*40,'c'*64)
    with pytest.raises(ValueError,match='source_digest'):
        validate_result('b5_bp__s0',summary,protocol,tasks,'a'*40,'d'*64)
    with pytest.raises(ValueError,match='synthetic'):
        validate_result('b5_bp__s0',summary,dict(protocol,synthetic=True),tasks)
    with pytest.raises(ValueError,match='trajectory'):
        validate_result('b5_bp__s0',dict(summary,tasks=tasks[:1]),protocol,tasks[:1])


def test_source_bundle_captures_runtime_overlay_without_outputs(tmp_path):
    import hashlib
    import zipfile
    from scripts.build_kaggle_vna_parallel import source_bundle
    path=tmp_path/'vna_v3_source.zip'
    digest=source_bundle(path)
    assert digest==hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        assert 'adapters/vna_v3_hooks.py' in names
        assert 'experiments/pilot_v3.py' in names
        assert not any(n.startswith(('notebooks/','.git/','outputs/')) or '/v3_logs/' in n for n in names)
