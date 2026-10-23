"""VNA output restore/archive helpers using the established Kaggle discovery."""
from pathlib import Path
import os
import shutil
import tarfile
import tempfile
import zipfile


def archive_outputs(output, budget=5_000_000_000, *, folders=("recipe", "signal"), exclude_parts=()):
    """Hard-link atomic immutable outputs before tar creation, even during training."""
    output = Path(output)
    target = output.with_suffix(".tar.gz")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="vna_snapshot_") as temporary:
        snapshot = Path(temporary)
        for folder in folders:
            for path in (output/folder).rglob("*"):
                if (path.is_file() and not path.name.endswith(".tmp")
                        and not any(part.endswith(".lock") for part in path.parts)
                        and not set(path.relative_to(output).parts).intersection(exclude_parts)):
                    dest = snapshot/path.relative_to(output)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    os.link(path, dest)
        for log in output.glob("*.log"):
            shutil.copy2(log, snapshot/log.name)
        raw = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())
        # During atomic replacement old archive + temporary archive coexist.
        free = shutil.disk_usage(output).free
        if raw * 3 > budget or raw > free:
            sizes = sorted(((p.stat().st_size, str(p.relative_to(snapshot)))
                            for p in snapshot.rglob('*') if p.is_file()), reverse=True)[:8]
            message = (f'ARCHIVE STALE: raw={raw} peak estimate={raw*3} budget={budget} free={free}; '
                       f'largest files={sizes}; existing archive retained')
            print(message, flush=True)
            raise RuntimeError(message)
        pending = target.with_suffix(".gz.tmp")
        try:
            with tarfile.open(pending, "w:gz", compresslevel=1) as archive:
                archive.add(snapshot, arcname=".")
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)
    print("Disk bytes:", raw, "archive bytes:", target.stat().st_size,
          "free bytes:", shutil.disk_usage(output).free, flush=True)
    return target


def restore_stage_archives(input_root, output, stage, *, exclude_parts=()):
    """Expanded files win over their own older archive; other conflicts refuse."""
    input_root, output = Path(input_root), Path(output)
    paths = sorted(p for p in input_root.rglob('*') if p.is_file())
    loose = {}
    for path in paths:
        if stage in path.parts and not set(path.parts[path.parts.index(stage)+1:]).intersection(exclude_parts):
            relative = Path(*path.parts[path.parts.index(stage)+1:])
            attachment = path.relative_to(input_root).parts[0]
            loose[(attachment, relative)] = path
    def save(name, data, attachment, archived=False):
        parts = Path(name).parts
        if '..' in parts or Path(name).is_absolute(): raise ValueError('unsafe archive member')
        if stage not in parts: return
        relative = Path(*parts[parts.index(stage)+1:])
        if not relative.parts or relative.name.endswith('.tmp') or set(relative.parts).intersection(exclude_parts): return
        if archived and (attachment, relative) in loose: return
        target = output/relative
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'conflicting {stage} restore: {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+'.tmp')
        temporary.write_bytes(data); temporary.replace(target)
    for path in paths:
        attachment = path.relative_to(input_root).parts[0]
        if stage in path.parts and not set(path.parts[path.parts.index(stage)+1:]).intersection(exclude_parts):
            save('/'.join(path.parts[path.parts.index(stage):]), path.read_bytes(), attachment)
        elif path.name.endswith(('.tar.gz', '.tar', '.tgz')):
            with tarfile.open(path) as archive:
                for member in archive:
                    if (member.isfile() and stage in Path(member.name).parts and
                        not set(Path(member.name).parts[Path(member.name).parts.index(stage)+1:]).intersection(exclude_parts)):
                        save(member.name, archive.extractfile(member).read(), attachment, True)
        elif path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if (not name.endswith('/') and stage in Path(name).parts and
                        not set(Path(name).parts[Path(name).parts.index(stage)+1:]).intersection(exclude_parts)):
                        save(name, archive.read(name), attachment, True)
