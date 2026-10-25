import tarfile
import pytest
from experiments import vna_kaggle


def test_archive_retries_atomic_replacement_race(tmp_path,monkeypatch):
    root=tmp_path/'run'
    folder=root/'outputs';folder.mkdir(parents=True)
    path=folder/'time_estimate.json';path.write_text('{"revision":1}')
    original=vna_kaggle.os.link
    calls=[]
    def replace_once(source,dest):
        calls.append(source)
        if len(calls)==1:
            pending=path.with_suffix('.tmp');pending.write_text('{"revision":2}');pending.replace(path)
            raise FileNotFoundError(str(source))
        return original(source,dest)
    monkeypatch.setattr(vna_kaggle.os,'link',replace_once)
    archive=vna_kaggle.archive_outputs(root,folders=('outputs',))
    with tarfile.open(archive) as stream:
        assert stream.extractfile('./outputs/time_estimate.json').read()==b'{"revision":2}'
    assert len(calls)==2


@pytest.mark.parametrize('archived',[False,True])
def test_restore_follows_mount_symlinks_preserves_baselines_and_exclusions(tmp_path,archived):
    mount=tmp_path/'mount';stage=mount/'outputs'/'p1_compare_nb1'
    baseline=stage/'icarl__s0';baseline.mkdir(parents=True)
    (baseline/'summary.json').write_bytes(b'original baseline')
    full=stage/'icarl_full__s0';full.mkdir()
    (full/'latest.pt').write_bytes(b'old full model')
    inputs=tmp_path/'input';inputs.mkdir()
    attachment=tmp_path/'archive' if archived else mount
    if archived:
        attachment.mkdir()
        with tarfile.open(attachment/'outputs.tar.gz','w:gz') as stream:
            stream.add(stage,arcname='outputs/p1_compare_nb1')
    (inputs/'nb1').symlink_to(attachment,target_is_directory=True)
    (attachment/'loop').symlink_to(inputs,target_is_directory=True)
    target=tmp_path/'restored'
    vna_kaggle.restore_stage_archives(inputs,target,'p1_compare_nb1',exclude_parts=('icarl_full__s0',))
    assert (target/'icarl__s0'/'summary.json').read_bytes()==b'original baseline'
    assert not (target/'icarl_full__s0').exists()
