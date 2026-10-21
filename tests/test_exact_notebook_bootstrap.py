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


@pytest.mark.parametrize('mode',['nan','profile'])
def test_historical_preparation_repairs_shared_partial_checkout(tmp_path,mode):
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
    source=source.replace('/kaggle/input',str(inputs)).replace('d88abc496eb1fe21d1b6df66d90b47cf93d7ded1',old).replace('8c56d02dab1ee16c47eb9138bf4aa3a64b9b45e2',old)
    scope=dict(MODE=mode,REPO=repo,Path=Path,subprocess=subprocess,shutil=shutil,private_clone=clone,MAIN_URL='fixture')
    exec(source,scope)
    assert (destination/'value.txt').read_text() == 'old solver'
    assert not (destination/'.git/objects/info/alternates').exists()
    assert (destination/'third_party/PyCIL').resolve() == repo/'third_party/PyCIL'
    exec(source,scope)
    assert clones == [old]*(2 if mode=='profile' else 1)
    if mode=='profile':
        reference=repo.with_name('counterfactual-projection-profile-reference')
        assert (reference/'value.txt').read_text()=='old solver'
        assert (reference/'third_party/PyCIL').resolve()==repo/'third_party/PyCIL'


def test_import_origins_reject_current_solver_for_old_checkout(tmp_path):
    old=tmp_path/'old';current=tmp_path/'current'
    main=SimpleNamespace(__name__='experiments.main_cil',__file__=str(old/'experiments/main_cil.py'))
    solver=SimpleNamespace(__name__='methods.vna_v3',__file__=str(current/'methods/vna_v3.py'))
    with pytest.raises(RuntimeError,match='Diagnostic import mismatch: methods.vna_v3'):
        verify_repository(old,(main,solver))
    solver.__file__=str(old/'methods/vna_v3.py')
    assert set(verify_repository(old,(main,solver))) == {'experiments.main_cil','methods.vna_v3'}


def checkpoint_bytes(**updates):
    import io
    import torch
    state=dict(identity=dict(arm='icarl_full',seed=0,backbone='resnet32_nobn',
        class_order_kind='superclass',nobn_lr_scale=.5,
        code_commit='d88abc496eb1fe21d1b6df66d90b47cf93d7ded1'),
        completed=1,mid=False,epoch=100,probe=torch.tensor([1.]))
    state.update(updates)
    data=io.BytesIO();torch.save(state,data)
    return data.getvalue()


def test_checkpoint_discovery_follows_inputs_and_deduplicates_formats(tmp_path):
    import io
    import tarfile
    import zipfile
    from diagnostics.exact_solver import find_task1_checkpoint
    from diagnostics.vna_align import _assert_bitwise
    from experiments.vna_resume import rng_state
    inputs=tmp_path/'input';inputs.mkdir()
    dataset=tmp_path/'mounted-dataset';dataset.mkdir()
    (inputs/'dataset').symlink_to(dataset,target_is_directory=True)
    (dataset/'loop').symlink_to(inputs,target_is_directory=True)
    payload=checkpoint_bytes()
    (dataset/'renamed-checkpoint.pth').write_bytes(payload)
    with tarfile.open(dataset/'unrelated-name.TGZ','w:gz') as archive:
        for name,data in [('snapshot/task1.pth',payload),('full/latest.previous.pt',checkpoint_bytes(mid=True))]:
            info=tarfile.TarInfo(name);info.size=len(data)
            archive.addfile(info,io.BytesIO(data))
    with zipfile.ZipFile(dataset/'another-output.zip','w') as archive:
        archive.writestr('custom-name/checkpoint.pt',payload)
    before=rng_state()
    selected=find_task1_checkpoint(inputs,tmp_path/'working/task1_source.pt')
    assert selected.read_bytes()==payload
    _assert_bitwise(before,rng_state())


def test_checkpoint_discovery_rejects_different_valid_states(tmp_path):
    import torch
    from diagnostics.exact_solver import find_task1_checkpoint
    inputs=tmp_path/'input';inputs.mkdir()
    (inputs/'one.pt').write_bytes(checkpoint_bytes())
    (inputs/'two.pt').write_bytes(checkpoint_bytes(probe=torch.tensor([2.])))
    target=tmp_path/'selected.pt'
    with pytest.raises(ValueError,match='Different compatible task-1 checkpoints'):
        find_task1_checkpoint(inputs,target)
    assert not target.exists()


def test_checkpoint_discovery_restores_cached_torch_utils_binding(tmp_path, monkeypatch):
    import io
    import torch
    from diagnostics.exact_solver import find_task1_checkpoint
    from diagnostics.vna_align import _assert_bitwise
    from experiments.vna_resume import rng_state
    payload=checkpoint_bytes()
    source=tmp_path/'latest.pt';source.write_bytes(payload)
    utils=torch._utils
    monkeypatch.delattr(torch, '_utils')
    with pytest.raises(AttributeError,match='_utils'):
        torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    before=rng_state()
    selected=find_task1_checkpoint(source,tmp_path/'task1_source.pt')
    assert torch._utils is utils and selected.read_bytes()==payload
    _assert_bitwise(before,rng_state())


def test_checkpoint_discovery_explains_missing_compatible_source(tmp_path):
    from diagnostics.exact_solver import find_task1_checkpoint
    wrong=tmp_path/'other-seed.pt'
    wrong.write_bytes(checkpoint_bytes(identity=dict(seed=1)))
    with pytest.raises(FileNotFoundError,match='other-seed.pt: metadata does not match'):
        find_task1_checkpoint(tmp_path,tmp_path/'target.pt')
