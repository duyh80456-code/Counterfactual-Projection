"""Kaggle may mount the archive itself or automatically extracted members."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
import pytest
from scripts.build_kaggle_vna_parallel import OVERLAY
from scripts.build_kaggle_p1_compare import build


@pytest.mark.parametrize('mode',['zip','extracted','modified','missing','symlink_zip','symlink_extracted'])
def test_overlay_verifies_the_pinned_zip_or_exact_members(tmp_path,monkeypatch,mode):
    monkeypatch.delenv('VNA_SOURCE_DIGEST',raising=False)
    inputs=tmp_path/'input';inputs.mkdir()
    repo=tmp_path/'repo';repo.mkdir()
    output=tmp_path/'output';output.mkdir()
    members={'pyproject.toml':b'[project]\n','experiments/main_cil.py':b'PINNED = True\n'}
    bundle=tmp_path/'vna_v3_source.zip'
    with zipfile.ZipFile(bundle,'w') as archive:
        for name,data in members.items():archive.writestr(name,data)
    digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in members.items()}
    if mode=='symlink_zip':
        mount=tmp_path/'mount';mount.mkdir()
        shutil.copy2(bundle,mount/bundle.name)
        (inputs/'source').symlink_to(mount,target_is_directory=True)
    elif mode=='zip':shutil.copy2(bundle,inputs/bundle.name)
    else:
        root=tmp_path/'mount' if mode=='symlink_extracted' else inputs/'datasets'/'source'
        root.mkdir(parents=True)
        if mode=='symlink_extracted':
            (inputs/'source').symlink_to(root,target_is_directory=True)
            (root/'loop').symlink_to(inputs,target_is_directory=True)
        with zipfile.ZipFile(bundle) as archive:archive.extractall(root)
        if mode=='modified':(root/'experiments/main_cil.py').write_bytes(b'changed')
        if mode=='missing':(root/'experiments/main_cil.py').unlink()
    scope=dict(SOURCE_DIGEST=digest,SOURCE_FILE_HASHES=manifest,REPO_COMMIT='a'*40,
               REPO=repo,RUN_ROOT=output,Path=Path,hashlib=hashlib,json=json,os=os,
               shutil=shutil,zipfile=zipfile)
    code=OVERLAY.replace('/kaggle/input',str(inputs))
    if mode in ('modified','missing'):
        with pytest.raises(FileNotFoundError):exec(code,scope)
        assert not (repo/'experiments/main_cil.py').exists()
        assert 'VNA_SOURCE_DIGEST' not in os.environ
        return
    exec(code,scope)
    for name,data in members.items():assert (repo/name).read_bytes()==data
    assert os.environ['VNA_SOURCE_DIGEST']==digest
    provenance=json.loads((output/'source_manifest.json').read_text())
    assert provenance['source_digest']==digest
    assert provenance['verification']==('zip_sha256' if mode in ('zip','symlink_zip') else 'extracted_member_sha256')
    notebook=build('a'*40,'NB1',digest,manifest)
    config=''.join(notebook['cells'][1]['source'])
    assert repr(manifest) in config and digest in config
