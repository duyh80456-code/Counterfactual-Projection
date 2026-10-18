"""Fail-closed validation for parallel experiment artifacts."""
import copy
import hashlib
import json
import re
from pathlib import Path


def dataset_fingerprint(root):
    result = {}
    for name in ('train', 'test', 'meta'):
        path = Path(root)/'cifar-100-python'/name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def validate_record(job, summary, protocol, tasks, expected_commit=None):
    if summary.get('status') != 'COMPLETED': raise ValueError(f'{job}: incomplete result')
    commit = protocol.get('code_commit')
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError(f'{job}: missing immutable code_commit')
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(f'{job}: code_commit mismatch')
    identity = summary.get('identity', summary)
    if identity.get('code_commit', commit) != commit:
        raise ValueError(f'{job}: summary/protocol code_commit mismatch')
    if protocol.get('benchmark') == 'v4_competitor':
        import math
        from experiments.v4_competitors import recipe, grid, ROUND2, ROUND3, SOURCES
        from adapters.v4_competitors import PINS
        from experiments.runners.pycil import UPSTREAM_PIN
        method = protocol.get('arm')
        backbone, order, seed = (protocol.get(k) for k in ('backbone','order','seed'))
        expected_job = f'{method}__{backbone}__{order}__s{seed}'
        if job != expected_job or method not in ('icarl',)+ROUND2+ROUND3:
            raise ValueError(f'{job}: invalid v4 competitor label')
        if backbone not in ('bn','nobn') or order not in ('superclass','random') or protocol.get('nobn_lr_scale',1.) not in (1.,.5):
            raise ValueError(f'{job}: invalid v4 setup')
        if protocol.get('fake_data') is not False or protocol.get('disabled') is not False:
            raise ValueError(f'{job}: synthetic or disabled competitor')
        cfg, source, overrides = recipe(method,backbone,seed,'cpu' if protocol.get('cfg',{}).get('device') == [-1] else 'cuda:0',
            protocol.get('reproduction',False),protocol.get('nobn_lr_scale',1.))
        if (identity != protocol or protocol.get('effective_method') != method or
                protocol.get('cfg') != cfg or protocol.get('cfg_source') != source or
                protocol.get('config_overrides') != overrides or protocol.get('upstream_commit') != UPSTREAM_PIN or
                protocol.get('competitor_pins') != PINS):
            raise ValueError(f'{job}: v4 recipe/provenance mismatch')
        if method in ROUND2 and protocol.get('method_config') not in grid(method):
            raise ValueError(f'{job}: unregistered competitor configuration')
        expected_impl = SOURCES.get(method,dict(source=f'PyCIL/models/{method}.py',reimplemented=False))
        if protocol.get('implementation') != expected_impl:
            raise ValueError(f'{job}: implementation provenance mismatch')
        if protocol.get('expected_tasks') != 20 or protocol.get('epochs') != ('200/170' if protocol.get('reproduction') else '100/50'):
            raise ValueError(f'{job}: v4 epoch/task policy mismatch')
        if summary.get('tasks') != tasks or len(tasks) != 20 or [r.get('task') for r in tasks] != list(range(1,21)):
            raise ValueError(f'{job}: incomplete v4 competitor tasks')
        for i,row in enumerate(tasks,1):
            if len(row.get('cnn_per_class',[])) != 5*i or any(not math.isfinite(row.get(k,float('nan'))) for k in
                    ('cnn_top1','cnn_new_top1','cnn_average_accuracy','cnn_forgetting','cnn_learning_accuracy')):
                raise ValueError(f'{job}: incomplete/nonfinite v4 metrics')
        fingerprint = protocol.get('dataset_fingerprint')
        if (not isinstance(fingerprint,dict) or set(fingerprint) != {'train','test','meta'} or
                any(not re.fullmatch('[0-9a-f]{64}',str(v)) for v in fingerprint.values())):
            raise ValueError(f'{job}: missing v4 dataset provenance')
        if summary.get('parameter_count',0) <= 0 or summary.get('gpu_hours',-1) < 0:
            raise ValueError(f'{job}: missing v4 resource accounting')
        return commit, ('v4_competitor',method,backbone,order), {
            'dataset':fingerprint,'upstream':UPSTREAM_PIN,'pins':PINS,'config':protocol['method_config']}
    if protocol.get('benchmark') == 'cil_competitor':
        from experiments.competitors import cil_recipe, CIL_METHODS
        from experiments.runners.pycil import UPSTREAM_PIN
        match = re.fullmatch(r'('+ '|'.join(CIL_METHODS)+r')__(?:(inc10|b50inc10)__)?s(\d+)', job)
        if not match: raise ValueError(f'{job}: invalid competitor label')
        method, setting, seed = match.groups()
        expected, source, overrides = cil_recipe(method, setting or 'inc10', int(seed))
        if identity != protocol or protocol.get('args') != expected or protocol.get('config_overrides') != overrides:
            raise ValueError(f'{job}: competitor recipe or override audit mismatch')
        if protocol.get('cfg_source') != source or protocol.get('upstream_commit') != UPSTREAM_PIN:
            raise ValueError(f'{job}: competitor cfg/upstream mismatch')
        if protocol.get('fake_data') is not False or protocol.get('arm') != method:
            raise ValueError(f'{job}: synthetic or mismatched competitor')
        expected_tasks = 1 if method == 'joint' else 1+(100-expected['init_cls'])//expected['increment']
        if (summary.get('tasks') != tasks or len(tasks) != expected_tasks or
                [r.get('task') for r in tasks] != list(range(1,expected_tasks+1))):
            raise ValueError(f'{job}: incomplete competitor tasks')
        fingerprint = protocol.get('dataset_fingerprint')
        if (not isinstance(fingerprint,dict) or set(fingerprint) != {'train','test','meta'} or
                any(not re.fullmatch('[0-9a-f]{64}',str(v)) for v in fingerprint.values())):
            raise ValueError(f'{job}: missing competitor data provenance')
        if summary.get('parameter_count',0) <= 0 or summary.get('gpu_hours',-1) < 0:
            raise ValueError(f'{job}: missing competitor resource accounting')
        return commit, ('cil_competitor',method,expected['init_cls'],expected['increment']), {
            'dataset':fingerprint, 'upstream':protocol['upstream_commit']}
    if protocol.get('benchmark') == 'b5_competitor':
        from experiments.runners import lop
        from experiments.competitors import b5_recipe
        if job != 'b5_'+protocol.get('arm','') or identity != protocol:
            raise ValueError(f'{job}: mismatched B5 competitor')
        variants = summary.get('variants',[])
        if protocol.get('fake_data') is not False or [r.get('seed') for r in variants] != [0,1,2]:
            raise ValueError(f'{job}: synthetic or incomplete competitor seeds')
        if protocol.get('upstream_commit') != lop.UPSTREAM_PIN:
            raise ValueError(f'{job}: wrong competitor upstream')
        for variant in variants:
            ident = variant.get('identity',{})
            expected, source, grid = b5_recipe(protocol['arm'],protocol['selected'])
            actual = dict(ident.get('params',{}))
            for key in ('data_file','env_file','run_idx','use_gpu'): actual.pop(key,None)
            if actual != expected or ident.get('code_commit') != commit or ident.get('fake_data') is not False:
                raise ValueError(f'{job}: B5 competitor recipe mismatch')
            if (variant.get('status') != 'COMPLETED' or variant.get('steps') != expected['num_data_points'] or
                    len(variant.get('tasks',[])) != expected['num_data_points']//expected['flip_after'] or
                    tasks.get(str(variant['seed'])) != variant['tasks']):
                raise ValueError(f'{job}: incomplete B5 competitor trajectory')
        return commit, ('b5_competitor',protocol['arm']), {'competitor':protocol['arm']}
    if job.startswith('icarl'):
        match = re.fullmatch(r'(icarl(?:_vna(?:_random|_euclid)?|_ngd)?)__(inc10|inc5|b50inc10)__s(\d+)', job)
        if not match: raise ValueError(f'{job}: unknown CIL job')
        label, setting, seed = match.groups()
        init, increment = (50,10) if setting == 'b50inc10' else (10,5 if setting == 'inc5' else 10)
        arm = 'icarl_vna' if label.startswith('icarl_vna') else label
        for key, expected in dict(seed=int(seed), init_cls=init, increment=increment, arm=arm).items():
            if protocol.get(key) != expected or summary.get(key) != expected:
                raise ValueError(f'{job}: {key} mismatch')
        for key in ('fake_data','smoke','pilot'):
            if protocol.get(key) is not False or summary.get(key) is not False:
                raise ValueError(f'{job}: {key} must explicitly be false')
        if protocol.get('upstream_commit') != identity.get('upstream_commit'):
            raise ValueError(f'{job}: upstream mismatch')
        if not protocol.get('upstream_commit'): raise ValueError(f'{job}: missing upstream commit')
        if identity.get('init_seed') != 0 or protocol.get('workers') is None:
            raise ValueError(f'{job}: missing/incompatible initialization or loader config')
        if arm != 'icarl':
            expected_selection = ('random_frob' if label.endswith('_random') else
                                  'tiny_V_euclid' if label.endswith('_euclid') else 'tiny_V')
            if (identity.get('intervention') or {}).get('selection') != expected_selection:
                raise ValueError(f'{job}: selection mismatch')
        fingerprint = protocol.get('dataset_fingerprint')
        if not isinstance(fingerprint, dict) or set(fingerprint) != {'train','test','meta'}:
            raise ValueError(f'{job}: missing dataset fingerprint')
        if any(not re.fullmatch('[0-9a-f]{64}', str(v)) for v in fingerprint.values()):
            raise ValueError(f'{job}: invalid dataset fingerprint')
        if identity.get('dataset_fingerprint') != fingerprint:
            raise ValueError(f'{job}: summary/protocol dataset mismatch')
        recipe = protocol.get('args')
        if not isinstance(recipe, dict) or identity.get('recipe') != recipe:
            raise ValueError(f'{job}: summary/protocol recipe mismatch')
        if recipe.get('dataset') != 'cifar100' or recipe.get('init_cls') != init or recipe.get('increment') != increment:
            raise ValueError(f'{job}: dataset/protocol mismatch')
        expected_tasks = 1 + (100-init+increment-1)//increment
        if (not isinstance(tasks, list) or len(tasks) != expected_tasks or
                summary.get('tasks_run') != expected_tasks or
                [r.get('task') for r in tasks] != list(range(1,expected_tasks+1))):
            raise ValueError(f'{job}: missing or unordered completed tasks')
        if summary.get('tasks') != tasks: raise ValueError(f'{job}: task artifacts disagree')
        return commit, ('cil',init,increment), {
            'recipe': {k:v for k,v in recipe.items() if k not in ('seed','device')},
            'dataset': fingerprint, 'upstream': protocol['upstream_commit'],
            'init_seed': identity.get('init_seed'), 'workers': protocol.get('workers')}
    if not job.startswith('b5_'): raise ValueError(f'{job}: unknown job')
    variants = summary.get('variants') or [summary]
    match = re.fullmatch(r'b5_(.+)__s([012])', job)
    arms = {'bp':'bp','cbp':'cbp','cbp_gir':'cbp_gir','cbp_gir_randdir':'cbp_gir_randdir',
            'bp_vna':'bp_vna_l1','bp_ngd':'bp_ngd_l1','cbp_gir_vna':'cbp_gir_vna_l1'}
    if not match or arms.get(match[1]) != summary.get('arm') or int(match[2]) != summary.get('seed'):
        raise ValueError(f'{job}: B5 job identity mismatch')
    if summary.get('variants'):
        expected_rates = protocol.get('replacement_rates')
        if (not expected_rates or sorted(v.get('replacement_rate') for v in variants) != sorted(expected_rates)
                or len(variants) != len(expected_rates)):
            raise ValueError(f'{job}: missing or duplicated registered rate')
    elif summary.get('identity') != protocol:
        raise ValueError(f'{job}: B5 summary/protocol identity mismatch')
    signatures = []
    for variant in variants:
        ident = variant.get('identity', {})
        if variant.get('status') != 'COMPLETED' or ident.get('code_commit') != commit:
            raise ValueError(f'{job}: incomplete/mixed-commit B5 variant')
        if ident.get('fake_data') is not False or ident.get('smoke') is not False:
            raise ValueError(f'{job}: synthetic B5 result')
        if variant.get('seed') != summary['seed'] or variant.get('arm') != summary['arm']:
            raise ValueError(f'{job}: variant identity mismatch')
        params = copy.deepcopy(ident.get('params', {}))
        if not params: raise ValueError(f'{job}: missing B5 parameters')
        expected_tasks = params['num_data_points']//params['flip_after']
        variant_tasks = variant.get('tasks', [])
        if (len(variant_tasks) != expected_tasks or variant.get('steps') != params['num_data_points']):
            raise ValueError(f'{job}: incomplete B5 trajectory')
        stored_tasks = tasks.get(str(variant['replacement_rate'])) if isinstance(tasks,dict) else tasks
        if stored_tasks != variant_tasks:
            raise ValueError(f'{job}: B5 task artifacts disagree')
        for key in ('data_file','env_file','replacement_rate','run_idx','agent','data_dir',
                    'num_runs','util_type','decay_rate','maturity_threshold'):
            params.pop(key, None)
        for key in list(params):
            if key.startswith('_comment'): params.pop(key)
        signatures.append(params)
    if any(s != signatures[0] for s in signatures):
        raise ValueError(f'{job}: variants disagree on training protocol')
    return commit, ('b5',), signatures[0]


def validate_baseline(directory, expected_commit, data_root):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text())
    protocol = json.loads((directory/'protocol.json').read_text())
    tasks = json.loads((directory/'tasks.json').read_text())
    validate_record('icarl__inc10__s1993', summary, protocol, tasks, expected_commit)
    from experiments.runners.pycil import recipe, UPSTREAM_PIN
    expected = recipe()
    actual = dict(protocol['args'])
    for key in ('device','seed'):
        expected.pop(key, None); actual.pop(key, None)
    if actual != expected or protocol['upstream_commit'] != UPSTREAM_PIN:
        raise ValueError('baseline recipe/upstream does not match the current experiment')
    if summary['identity'].get('init_seed') != 0:
        raise ValueError('baseline initialization seed mismatch')
    if protocol['dataset_fingerprint'] != dataset_fingerprint(data_root):
        raise ValueError('baseline dataset content mismatch')
    return summary
