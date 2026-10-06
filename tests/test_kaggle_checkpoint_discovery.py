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


def test_truncated_gzip_is_reported_without_losing_other_forks(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    checkpoint = input_root / "valid.pt"
    torch.save(payload(), checkpoint)
    archive_path = input_root / "broken.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(checkpoint, arcname="run/valid.pt")
    archive_path.write_bytes(archive_path.read_bytes()[:20])

    matches, rejected = discover_checkpoints(
        input_root, tmp_path / "output", kind="plateau_fork_checkpoint")
    assert len(matches) == 1
    error = next(item for item in rejected if item["path"] == str(archive_path))
    assert error["reason"] in {"EOFError", "ReadError"}
    assert error["detail"]


def test_recovers_complete_checkpoint_before_truncated_archive_tail(tmp_path):
    import os
    import io

    checkpoint = tmp_path / "source.pt"
    torch.save(payload(), checkpoint)
    input_root = tmp_path / "input"
    input_root.mkdir()
    archive_path = input_root / "broken-tail.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(checkpoint, arcname="run/plateau_checkpoint.pt")
        info = tarfile.TarInfo("large-unrelated.bin")
        data = os.urandom(2 * 1024 * 1024)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    compressed = archive_path.read_bytes()
    archive_path.write_bytes(compressed[:len(compressed) // 2])

    matches, rejected = discover_checkpoints(
        input_root, tmp_path / "output", kind="plateau_fork_checkpoint")
    assert len(matches) == 1
    assert matches[0]["payload"]["epoch"] == 328
    error = next(item for item in rejected if item["path"] == str(archive_path))
    assert error["complete_checkpoints_recovered"] == 1


def test_incomplete_checkpoint_member_is_not_offered_for_loading(tmp_path):
    source = tmp_path / "source.pt"
    torch.save({**payload(), "large": torch.rand(500_000)}, source)
    input_root = tmp_path / "input"
    input_root.mkdir()
    archive_path = input_root / "incomplete-checkpoint.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname="plateau_checkpoint.pt")
    data = archive_path.read_bytes()
    archive_path.write_bytes(data[:len(data) // 2])
    matches, rejected = discover_checkpoints(
        input_root, tmp_path / "output", kind="plateau_fork_checkpoint")
    assert matches == []
    assert rejected[0]["complete_checkpoints_recovered"] == 0
    assert not list((tmp_path / "output").rglob("*.pt"))
