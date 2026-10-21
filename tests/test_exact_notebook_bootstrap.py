"""Historical checkout must be independent, complete and imported from the selected tree."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import pytest
from diagnostics.exact_solver import verify_repository
from scripts.build_kaggle_pilot_common import BOOTSTRAP
from scripts.build_kaggle_vna_parallel import EXACT_DIAGNOSTICS


def test_private_clone_authenticates_historical_checkout(tmp_path):
    node = next(n for n in ast.parse(BOOTSTRAP).body if isinstance(n, ast.FunctionDef) and n.name == 'private_clone')
    calls = []
    askpass = tmp_path/'askpass.py'
    secret = SimpleNamespace(get_secret=lambda name: 'test-placeholder')
    scope = dict(UserSecretsClient=lambda: secret, os=os, REPO_COMMIT='a'*40,
                 Path=lambda ignored: askpass,
                 subprocess=SimpleNamespace(run=lambda command, **kw: calls.append((command, kw))))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<private_clone>', 'exec'), scope)
    scope['private_clone']('test://repo', tmp_path/'old', 'b'*40)
    assert [c[0][1] if c[0][1] == 'clone' else c[0][3] for c in calls] == ['clone','fetch','checkout']
    assert calls[1][0][-1] == calls[2][0][-1] == 'b'*40
    assert all(c[1]['check'] and c[1]['env']['GIT_ASKPASS'] == str(askpass) for c in calls)
    assert not askpass.exists()


def test_historical_preparation_repairs_shared_partial_checkout(tmp_path):
    repo = tmp_path/'current'
    repo.mkdir()
    def git(*args, cwd=repo):
        return subprocess.check_output(['git', *args], cwd=cwd, text=True, stderr=subprocess.PIPE).strip()
    git('init', '-b', 'main')
    (repo/'third_party').mkdir()
    (repo/'third_party/.gitkeep').touch()
    (repo/'value.txt').write_text('old solver')
    git('add', '.')
    def commit(message):
        git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
            '-c','commit.gpgsign=false','commit','-m',message)
        return git('rev-parse','HEAD')
    old = commit('historical')
    (repo/'value.txt').write_text('new solver')
    git('add','value.txt'); commit('current')
    (repo/'third_party/PyCIL').mkdir()
    destination = repo.with_name('counterfactual-projection-d88abc4')
    git('clone','--shared',str(repo),str(destination),cwd=tmp_path)
    (destination/'value.txt').unlink()
    inputs=tmp_path/'input';inputs.mkdir();(inputs/'vna4_main.tar.gz').touch()
    clones=[]
    def clone(url, target, revision):
        clones.append(revision)
        git('clone','--no-local',str(repo),str(target),cwd=tmp_path)
        git('checkout','--detach',revision,cwd=target)
    source=EXACT_DIAGNOSTICS[:EXACT_DIAGNOSTICS.index("script = REPO/")]
    source=source.replace('/kaggle/input',str(inputs)).replace('d88abc496eb1fe21d1b6df66d90b47cf93d7ded1',old)
    scope=dict(REPO=repo,Path=Path,subprocess=subprocess,shutil=shutil,private_clone=clone,MAIN_URL='fixture')
    exec(source,scope)
    assert (destination/'value.txt').read_text() == 'old solver'
    assert not (destination/'.git/objects/info/alternates').exists()
    assert (destination/'third_party/PyCIL').resolve() == repo/'third_party/PyCIL'
    exec(source,scope)
    assert clones == [old]


def test_import_origins_reject_current_solver_for_old_checkout(tmp_path):
    old=tmp_path/'old';current=tmp_path/'current'
    main=SimpleNamespace(__name__='experiments.main_cil',__file__=str(old/'experiments/main_cil.py'))
    solver=SimpleNamespace(__name__='methods.vna_v3',__file__=str(current/'methods/vna_v3.py'))
    with pytest.raises(RuntimeError,match='Diagnostic import mismatch: methods.vna_v3'):
        verify_repository(old,(main,solver))
    solver.__file__=str(old/'methods/vna_v3.py')
    assert set(verify_repository(old,(main,solver))) == {'experiments.main_cil','methods.vna_v3'}
