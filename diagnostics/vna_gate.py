"""Pre-registered Stage-B gate; pure logic, no training or file I/O."""
RHO_MIN = 0.3
RHO_FRAC = 0.5
N1_FRAC = 0.5
TINYV_FRAC = 0.7
TRAIN_ACC_MAX = 0.99
GATE_RANK = 16


def setting_config(setting, *, tasks_run=None, mu_rel_default=1e-2):
    if setting in ('B1', 'B2'):
        window = lambda r: r['epoch'] < 210
    elif setting == 'B4':
        window = lambda r: r['epoch'] < 150
    elif setting == 'B3':
        window = lambda r: r['task'] >= 2 and not r['end_of_task']
    elif setting == 'B5':
        if tasks_run is None:
            raise ValueError('B5 requires tasks_run')
        window = lambda r: r['task'] > tasks_run / 2
    else:
        raise ValueError(f'unknown setting {setting}')
    return {'window': window, 'train_acc_applicable': setting not in ('B3', 'B5'),
            'mu_rel_default': mu_rel_default, 'damping_norm': 'outputs'}


BUILTIN_CONFIGS = {s: setting_config(s) for s in ('B1', 'B2', 'B3', 'B4')}


def evaluate_gate(rows, setting_cfg):
    rows = list(rows)
    candidates = [r for r in rows if r['rank'] == setting_cfg.get('gate_rank', GATE_RANK)
                  and r.get('damping_norm', 'outputs') == setting_cfg.get('damping_norm', 'outputs')
                  and r['mu_rel'] == setting_cfg.get('mu_rel_default', 1e-2) and setting_cfg['window'](r)]
    # The unit of exclusion is a measurement point, including all its sites.
    point_key = lambda r: tuple(r.get(k) for k in ('setting', 'benchmark', 'seed', 'task', 'epoch', 'step', 'point'))
    points = {point_key(r) for r in candidates}
    excluded = {point_key(r) for r in candidates
                if not r.get('cg_converged', False) or r.get('status') == 'timeout'}
    excluded_fraction = len(excluded)/len(points) if points else 0.
    eligible = [r for r in candidates if point_key(r) not in excluded]
    n = len(eligible)
    frac = lambda fn: sum(bool(fn(r)) for r in eligible) / n if n else 0.
    rho = frac(lambda r: r['rho_joint'] >= RHO_MIN)
    n1 = frac(lambda r: r['significant_rho_N1'])
    tiny = frac(lambda r: r['V_tiny_V'] > r['V_random_frob'])
    applicable = setting_cfg['train_acc_applicable']
    final = setting_cfg.get('final_train_acc')
    if final is None and rows:
        final = max(rows, key=lambda r: (r.get('task', 0), r.get('epoch', 0))).get('train_acc')
    conditions = {'rho': rho >= RHO_FRAC, 'N1': n1 >= N1_FRAC,
                  'tiny_V': tiny >= TINYV_FRAC,
                  'train_acc': (final is not None and final < TRAIN_ACC_MAX) if applicable else 'n/a'}
    return {'status': 'INVALID' if excluded_fraction > .10 else 'PASS' if n and all(v is True or v == 'n/a' for v in conditions.values()) else 'FAIL',
            'eligible_points': n, 'excluded_points': len(excluded),
            'excluded_fraction': excluded_fraction, 'gate_null_statistic': 'rho', 'rho_fraction': rho, 'N1_fraction': n1,
            'tiny_V_fraction': tiny, 'final_train_acc': final if applicable else 'n/a',
            'raw_final_train_acc': final, 'conditions': conditions}
