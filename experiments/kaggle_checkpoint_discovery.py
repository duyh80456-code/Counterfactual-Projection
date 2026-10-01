"""Discover ordinary or Kaggle-expanded PyTorch checkpoints."""

from __future__ import annotations

import hashlib
import os
import tarfile
import zipfile
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repack_archive(root: Path, target: Path) -> Path:
    """Rebuild a torch-save zip whose internal files Kaggle exposed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for record in sorted(root.rglob("*")):
            if record.is_file():
                relative = record.relative_to(root).as_posix()
                archive.writestr(f"checkpoint/{relative}", record.read_bytes())
    return target


def _extract_checkpoint_members(archive_path: Path, target_root: Path
                                ) -> list[Path]:
    """Materialize .pt members from a notebook-output zip/tar archive."""
    target_root.mkdir(parents=True, exist_ok=True)
    extracted = []
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                members = [name for name in archive.namelist()
                           if name.lower().endswith(".pt")]
                for index, name in enumerate(members):
                    target = target_root / f"zip_{index}_{Path(name).name}"
                    target.write_bytes(archive.read(name))
                    extracted.append(target)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                members = [member for member in archive.getmembers()
                           if member.isfile() and
                           member.name.lower().endswith(".pt")]
                for index, member in enumerate(members):
                    stream = archive.extractfile(member)
                    if stream is None:
                        continue
                    target = (
                        target_root / f"tar_{index}_{Path(member.name).name}")
                    target.write_bytes(stream.read())
                    extracted.append(target)
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        return []
    return extracted


def discover_checkpoints(input_root: str | Path, output: str | Path,
                         *, kind: str | set[str] | tuple[str, ...]
                         ) -> tuple[list[dict], list[dict]]:
    """Load every plausible checkpoint and retain requested payload kinds.

    Kaggle may expose a torch-save zip as a directory tree. Therefore discovery
    searches by payload type, not filename: ordinary ``*.pt`` files are tried
    directly and every directory containing ``data.pkl`` is repacked first.
    """
    wanted = {kind} if isinstance(kind, str) else set(kind)
    if not wanted:
        raise ValueError("at least one checkpoint kind is required")
    input_root = Path(input_root)
    output = Path(output)
    # Notebook outputs can be mounted below /kaggle/input through directory
    # symlinks. pathlib.rglob does not descend into those links, whereas
    # os.walk(..., followlinks=True) does.
    pt_files, data_pickles, container_archives = [], [], []
    for directory, _, filenames in os.walk(input_root, followlinks=True):
        root = Path(directory)
        for filename in filenames:
            path = root / filename
            if filename.endswith(".pt"):
                pt_files.append(path)
            if filename == "data.pkl":
                data_pickles.append(path)
            lowered = filename.lower()
            if (lowered.endswith((".zip", ".tar", ".tar.gz", ".tgz")) and
                    not lowered.endswith(".pt")):
                container_archives.append(path)
    candidates: list[tuple[Path, str]] = [
        (path, "file") for path in sorted(set(pt_files)) if path.is_file()
    ]
    archive_roots = sorted({path.parent.resolve() for path in data_pickles})
    for index, root in enumerate(archive_roots):
        target = output / "repacked_input" / f"torch_archive_{index}.pt"
        candidates.append((_repack_archive(root, target), str(root)))
    for index, archive_path in enumerate(sorted(set(container_archives))):
        extracted = _extract_checkpoint_members(
            archive_path, output / "repacked_input" /
            f"container_archive_{index}")
        candidates.extend((path, str(archive_path)) for path in extracted)

    matches, rejected = [], []
    seen_paths = set()
    for path, source in candidates:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            rejected.append({"path": str(path), "reason": type(error).__name__})
            continue
        found_kind = payload.get("kind") if isinstance(payload, dict) else None
        if found_kind not in wanted:
            rejected.append({"path": str(path), "kind": found_kind})
            continue
        matches.append({
            "path": path, "payload": payload, "sha256": _sha256(path),
            "source": source,
        })
    return matches, rejected
