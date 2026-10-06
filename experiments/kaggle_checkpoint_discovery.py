"""Discover ordinary or Kaggle-expanded PyTorch checkpoints."""

from __future__ import annotations

import hashlib
import errno
import os
import shutil
import tarfile
import zipfile
import zlib
from pathlib import Path

import torch


def _check_space(directory: Path, size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    reserve = 256 * 1024 * 1024
    if size + reserve > free:
        raise RuntimeError(
            f"Checkpoint discovery needs {size} bytes plus a {reserve}-byte "
            f"reserve at {directory}; only {free} bytes free. "
            "Start a fresh session or attach fewer checkpoint outputs.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repack_archive(root: Path, target: Path) -> Path:
    """Rebuild a torch-save zip whose internal files Kaggle exposed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    records = [record for record in sorted(root.rglob("*")) if record.is_file()]
    _check_space(target.parent, sum(record.stat().st_size for record in records))
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for record in records:
            relative = record.relative_to(root).as_posix()
            archive.write(record, f"checkpoint/{relative}")
    return target


def _extract_checkpoint_members(archive_path: Path, target_root: Path, *,
                                rejected: list[dict] | None = None,
                                consume=None) -> list[Path]:
    """Materialize .pt members from a notebook-output zip/tar archive."""
    target_root.mkdir(parents=True, exist_ok=True)
    extracted = []
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                members = [info for info in archive.infolist()
                           if info.filename.lower().endswith(".pt")]
                for index, info in enumerate(members):
                    target = target_root / f"zip_{index}_{Path(info.filename).name}"
                    _check_space(target_root, info.file_size)
                    temporary = target.with_suffix(".pt.tmp")
                    try:
                        with archive.open(info) as stream, temporary.open("wb") as destination:
                            shutil.copyfileobj(stream, destination)
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
                    if consume is None or consume(target, str(archive_path), True):
                        extracted.append(target)
        else:
            # Stream tar members instead of scanning the whole archive first.
            # Complete checkpoints before a truncated tail remain usable.
            with tarfile.open(archive_path, mode="r|*") as archive:
                for index, member in enumerate(archive):
                    if not member.isfile() or not member.name.lower().endswith(".pt"):
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        continue
                    target = (
                        target_root / f"tar_{index}_{Path(member.name).name}")
                    temporary = target.with_suffix(".pt.tmp")
                    _check_space(target_root, member.size)
                    try:
                        with stream, temporary.open("wb") as destination:
                            shutil.copyfileobj(stream, destination)
                        if temporary.stat().st_size != member.size:
                            raise EOFError(f"Incomplete checkpoint member: {member.name}")
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
                    if consume is None or consume(target, str(archive_path), True):
                        extracted.append(target)
    except (OSError, EOFError, tarfile.TarError, zipfile.BadZipFile, zlib.error) as error:
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            raise RuntimeError(f"Disk full during checkpoint discovery at {target_root}") from error
        if rejected is not None:
            rejected.append({"path": str(archive_path),
                             "reason": type(error).__name__,
                             "detail": str(error),
                             "complete_checkpoints_recovered": len(extracted)})
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
    matches, rejected = [], []
    seen_paths, seen_hashes = set(), set()

    def consume(path: Path, source: str, temporary: bool = False) -> bool:
        keep = False
        try:
            resolved = path.resolve()
            if resolved in seen_paths:
                return False
            seen_paths.add(resolved)
            digest = _sha256(path)
            if digest in seen_hashes:
                return False
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as error:
                rejected.append({"path": str(path), "reason": type(error).__name__})
                return False
            found_kind = payload.get("kind") if isinstance(payload, dict) else None
            if found_kind not in wanted:
                rejected.append({"path": str(path), "kind": found_kind})
                return False
            seen_hashes.add(digest)
            matches.append({"path": path, "payload": payload,
                            "sha256": digest, "source": source})
            keep = True
            return True
        finally:
            if temporary and not keep:
                path.unlink(missing_ok=True)

    # Process each candidate immediately. Only requested, unique checkpoints
    # remain on disk; unrelated arm checkpoints never accumulate in the cache.
    for path in sorted(set(pt_files)):
        if path.is_file():
            consume(path, "file")
    archive_roots = sorted({path.parent.resolve() for path in data_pickles})
    for index, root in enumerate(archive_roots):
        target = output / "repacked_input" / f"torch_archive_{index}.pt"
        consume(_repack_archive(root, target), str(root), True)
    for index, archive_path in enumerate(sorted(set(container_archives))):
        _extract_checkpoint_members(
            archive_path, output / "repacked_input" /
            f"container_archive_{index}", rejected=rejected, consume=consume)
    return matches, rejected
