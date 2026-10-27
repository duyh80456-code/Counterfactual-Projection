"""Reference inputs stay separate; resume of the selected run stays strict."""
import io
import tarfile
import zipfile
from pathlib import Path
import pytest
from experiments.vna_kaggle import restore_stage_archives


def test_namespaced_loose_sources_do_not_merge_shared_dataset_prefix(tmp_path):
    inputs=tmp_path/'input'
    for name,value in (('gram','old'),('main','new')):
        path=inputs/f'datasets/user/{name}/outputs/vna_v4_main/flag_rates.json'
        path.parent.mkdir(parents=True);path.write_text(value)
    out=tmp_path/'references'
    restore_stage_archives(inputs,out,'outputs',namespace=True)
    assert (out/'datasets/user/gram/vna_v4_main/flag_rates.json').read_text()=='old'
    assert (out/'datasets/user/main/vna_v4_main/flag_rates.json').read_text()=='new'
    with pytest.raises(ValueError,match='conflicting outputs restore'):
        restore_stage_archives(inputs,tmp_path/'merged','outputs')


def test_namespaced_archives_preserve_both_versions_and_resume_conflicts_refuse(tmp_path):
    inputs=tmp_path/'input';inputs.mkdir()
    for name,value in (('a.zip','first'),('b.zip','second')):
        with zipfile.ZipFile(inputs/name,'w') as archive:
            archive.writestr('outputs/vna_v4_main/flag_rates.json',value)
            archive.writestr('outputs/pin/run/checkpoint.pt',value)
    refs=tmp_path/'refs'
    restore_stage_archives(inputs,refs,'outputs',namespace=True)
    assert (refs/'a.zip/vna_v4_main/flag_rates.json').read_text()=='first'
    assert (refs/'b.zip/vna_v4_main/flag_rates.json').read_text()=='second'
    with pytest.raises(ValueError,match='pin/run/checkpoint.pt'):
        restore_stage_archives(inputs,tmp_path/'resume','outputs',prefix='pin/run')


def test_resume_prefix_ignores_unrelated_files_without_reading_them(tmp_path,monkeypatch):
    inputs=tmp_path/'input'
    unrelated=inputs/'dataset/outputs/vna_v4_main/flag_rates.json'
    unrelated.parent.mkdir(parents=True);unrelated.write_text('unrelated')
    selected=inputs/'dataset/outputs/pin/run/summary.json'
    selected.parent.mkdir(parents=True);selected.write_text('selected')
    read=Path.read_bytes
    def guarded(path):
        assert path!=unrelated
        return read(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    out=tmp_path/'out'
    restore_stage_archives(inputs,out,'outputs',prefix='pin/run')
    assert (out/'pin/run/summary.json').read_text()=='selected'
    assert not (out/'vna_v4_main').exists()


def test_default_expanded_file_still_wins_own_stale_tar_and_unsafe_members_refuse(tmp_path):
    inputs=tmp_path/'input';dataset=inputs/'dataset';dataset.mkdir(parents=True)
    path=dataset/'outputs/run/status.json';path.parent.mkdir(parents=True);path.write_text('new')
    with tarfile.open(dataset/'old.tar.gz','w:gz') as archive:
        member=tarfile.TarInfo('outputs/run/status.json');member.size=3
        archive.addfile(member,io.BytesIO(b'old'))
    out=tmp_path/'out'
    restore_stage_archives(inputs,out,'outputs')
    assert (out/'run/status.json').read_text()=='new'
    with zipfile.ZipFile(dataset/'unsafe.zip','w') as archive:
        archive.writestr('outputs/../escape','bad')
    with pytest.raises(ValueError,match='unsafe archive member'):
        restore_stage_archives(inputs,tmp_path/'unsafe','outputs',namespace=True)


def test_kernel_notebook_separates_references_from_selected_resume():
    from scripts.build_kaggle_kernel_ablation import build
    import subprocess
    nb=build(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    assert "REFERENCE_INPUTS, 'outputs', namespace=True" in code
    assert "STAGE_B, 'outputs', prefix=REPO_COMMIT[:12]+'/'+RUN_TAG" in code
    assert 'REFERENCE_INPUTS,STAGE_B' in code
