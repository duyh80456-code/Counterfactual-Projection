"""The new pilot accepts the actual main and competitor identity schemas."""
import ast
import copy
import json
from pathlib import Path
import pytest
from adapters.cil_setup import class_order
from experiments import main_cil as main,v4_competitors as competitors
from scripts.build_kaggle_vna5_inc10_p2 import RUNNER

COMMIT='a'*40


def notebook_functions(names,**values):
    nodes=[n for n in ast.parse(RUNNER).body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace=dict(REPO_COMMIT=COMMIT,SOURCE_DIGEST=None,**values)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pilot_runtime','exec'),namespace)
    return namespace


def capture_identity(monkeypatch,tmp_path,arm):
    main.cil.upstream_imports()
    fingerprint={name:'b'*64 for name in ('train','test','meta')}
    monkeypatch.delenv('VNA_SOURCE_DIGEST',raising=False)
    monkeypatch.setattr('experiments.parallel_validity.dataset_fingerprint',lambda root:fingerprint)
    order=class_order(0)
    monkeypatch.setattr(main,'class_order',lambda *a:order)
    monkeypatch.setattr(competitors,'class_order',lambda *a:order)
    monkeypatch.setattr(main,'code_commit',lambda:COMMIT)
    monkeypatch.setattr(competitors,'code_commit',lambda:COMMIT)
    seen={}
    class Captured(Exception):pass
    def progress(directory,identity):
        seen.update(copy.deepcopy(identity))
        raise Captured()
    common=['--out',str(tmp_path/arm),'--increment','10','--warmup-epochs','1','--seed','0','--device','cpu']
    if arm in ('icarl','icarl_full'):
        module=main
        cli=['--arm',arm,'--solver','gram','--curvature','power10','--max-tasks','10',*common]
        if arm=='icarl_full':
            schedule=tmp_path/'authority.json'
            schedule.write_text(json.dumps(dict(authoritative=True,code_commit=COMMIT,source_digest=None,
                backbone='resnet32_nobn',nobn_lr_scale=.5,curvature='power10',solver='gram',
                policy_version='certswap_t32_v1',gir_live=False,init_cls=10,increment=10,warmup_epochs=1,
                gir_rule='certswap',certswap_topk=8,certswap_lm_budget=20,certswap_log_schur=True,
                certswap_candidate_rule='tiny_V_unchanged_no_median_no_sign',K0=32)))
            cli+=['--gir-rule','certswap','--schedule',str(schedule)]
    else:
        module=competitors;cli=['--method',arm,*common]
    monkeypatch.setattr(module,'Progress',progress)
    with pytest.raises(Captured):module.run(module.parser().parse_args(cli))
    return seen


@pytest.mark.parametrize('arm',['icarl','icarl_full','icarl_cbp','icarl_redo'])
def test_notebook_identity_accepts_both_actual_runner_summary_schemas(monkeypatch,tmp_path,arm):
    protocol=capture_identity(monkeypatch,tmp_path,arm)
    assert protocol['cfg']['init_cls']==protocol['cfg']['increment']==10
    assert protocol['expected_tasks']==10 and protocol['training_protocol']=='P2'
    assert protocol['warmup_epochs']==1 and len(protocol['class_order_sha256'])==64
    summary=dict(status='COMPLETED',identity=protocol,arm=arm,seed=0)
    validate=notebook_functions({'identity'})['identity']
    validate(summary,arm)
    validate(protocol,arm)
    wrong=copy.deepcopy(summary);wrong['identity']['warmup_epochs']=0
    with pytest.raises(ValueError):validate(wrong,arm)
    wrong=copy.deepcopy(summary);wrong['identity']['cfg']['increment']=5
    with pytest.raises(ValueError):validate(wrong,arm)
    if arm=='icarl_full':
        assert protocol['policy_version']=='certswap_t32_v1'
        wrong=copy.deepcopy(summary);wrong['identity']['policy_version']='schur_gamma_bound_rho25_v2'
        with pytest.raises(ValueError):validate(wrong,arm)


def test_notebook_report_reads_main_and_competitor_metric_field_names(tmp_path):
    import numpy as np
    arms=('icarl_full','icarl','icarl_cbp','icarl_redo')
    for arm in arms:
        directory=tmp_path/(arm+'__nobn__superclass__inc10__p2__s0');directory.mkdir()
        row=dict(task=10)
        for label in ('cnn','nme'):
            row.update(main.accuracy_metrics([],np.array([[0],[1]]),np.array([0,1]),0,label))
        tasks=[dict(row,task=task) for task in range(1,11)]
        summary=dict(status='COMPLETED',tasks=tasks)
        for label in ('cnn','nme'):
            for name in ('average_accuracy','last_accuracy','learning_accuracy','forgetting'):
                summary[label+'_'+name]=row[label+'_'+name]
        (directory/'summary.json').write_text(json.dumps(summary))
    functions=notebook_functions({'read','job_name','report'},ROOT=tmp_path,ARMS=arms,
        json=json,report_lock=__import__('threading').Lock(),errors=[],
        atomic_json_save=lambda value,path:None)
    rows=functions['report']()['task10']
    assert len(rows)==4
    assert all(row['cnn_average']==row['nme_average']==row['cnn_last']==row['nme_last']==100 for row in rows)
    assert all(row['learning_accuracy']==row['nme_learning_accuracy']==100 and row['forgetting']==0 for row in rows)


def test_first_pilot_code_cell_gates_all_launches_on_three_unskipped_cuda_tests():
    from scripts.build_kaggle_vna5_inc10_p2 import build,GPU_GATE
    document=build(COMMIT)
    cells=[cell for cell in document['cells'] if cell['cell_type']=='code']
    first=''.join(cells[0]['source']);last=''.join(cells[-1]['source'])
    assert 'tests/test_certswap_cuda.py' in first
    assert first.index('private_clone(MAIN_URL')<first.index('tests/test_certswap_cuda.py')
    assert 'len(cases)!=3' in GPU_GATE and "'skipped','failure','error'" in GPU_GATE
    assert 'if gate.returncode:raise RuntimeError' in first
    assert "globals().get('GPU_GATE_PASSED',False)" in last
    assert last.index("globals().get('GPU_GATE_PASSED',False)")<last.index('def prepare_schedule')
