import tarfile
import zipfile

import torch

from experiments.kaggle_checkpoint_discovery import discover_checkpoints


def payload():
    return {"kind": "plateau_fork_checkpoint", "epoch": 328,
            "model": {"weight": torch.tensor([1.0])}}


def test_discovers_checkpoint_by_payload_kind_not_filename(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    torch.save(payload(), input_root / "some-renamed-upload.pt")
    matches, _ = discover_checkpoints(
        input_root, tmp_path / "output", kind="plateau_fork_checkpoint")
    assert len(matches) == 1
    assert matches[0]["payload"]["epoch"] == 328


def test_rebuilds_kaggle_expanded_torch_archive(tmp_path):
    source = tmp_path / "source.pt"
    torch.save(payload(), source)
    input_root = tmp_path / "input"
    input_root.mkdir()
    with zipfile.ZipFile(source) as archive:
        archive.extractall(input_root / "mounted-checkpoint-without-pt-name")

    matches, _ = discover_checkpoints(
        input_root, tmp_path / "output", kind="plateau_fork_checkpoint")
    assert len(matches) == 1
    assert matches[0]["payload"]["kind"] == "plateau_fork_checkpoint"


def test_follows_kaggle_notebook_output_directory_symlink(tmp_path):
    actual_output = tmp_path / "saved-notebook-output"
    actual_output.mkdir()
    torch.save(payload(), actual_output / "plateau_checkpoint.pt")
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "attached-notebook").symlink_to(
        actual_output, target_is_directory=True)

    matches, _ = discover_checkpoints(
        input_root, tmp_path / "repacked", kind="plateau_fork_checkpoint")
    assert len(matches) == 1
    assert matches[0]["payload"]["epoch"] == 328


def test_discovers_checkpoint_inside_notebook_output_tarball(tmp_path):
    checkpoint = tmp_path / "checkpoint_best.pt"
    torch.save(payload(), checkpoint)
    input_root = tmp_path / "input"
    input_root.mkdir()
    archive_path = input_root / "saved-run.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(checkpoint, arcname="run/checkpoint_best.pt")

    matches, _ = discover_checkpoints(
        input_root, tmp_path / "output", kind="plateau_fork_checkpoint")
    assert len(matches) == 1
    assert matches[0]["payload"]["epoch"] == 328
    assert matches[0]["source"] == str(archive_path)
