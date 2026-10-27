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
        if raw * 3 > budget:
            sizes = sorted(((p.stat().st_size, str(p.relative_to(snapshot)))
                            for p in snapshot.rglob('*') if p.is_file()), reverse=True)[:8]
            message = (f'ARCHIVE STALE: raw={raw} peak estimate={raw*3} budget={budget}; '
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


def restore_stage_archives(input_root, output, stage, *, namespace=False, prefix=None):
    """Keep optional reference sources separate; selected resume conflicts still refuse."""
    input_root, output = Path(input_root), Path(output)
    prefix = Path(prefix).parts if prefix is not None else ()
    if '..' in prefix or (prefix and Path(*prefix).is_absolute()):raise ValueError('unsafe restore prefix')
    paths = sorted(p for p in input_root.rglob('*') if p.is_file())
    def origin(path):
        relative = path.relative_to(input_root)
        if not namespace:return relative.parts[0]
        return str(Path(*relative.parts[:relative.parts.index(stage)])) if stage in relative.parts else str(relative)
    loose = {}
    for path in paths:
        if stage in path.parts:
            relative = Path(*path.parts[path.parts.index(stage)+1:])
            attachment = origin(path)
            loose[(attachment, relative)] = path
    def save(name, read_data, attachment, archived=False):
        parts = Path(name).parts
        if '..' in parts or Path(name).is_absolute(): raise ValueError('unsafe archive member')
        if stage not in parts: return
        relative = Path(*parts[parts.index(stage)+1:])
        if not relative.parts or relative.name.endswith('.tmp'): return
        if relative.parts[:len(prefix)]!=prefix:return
        if archived and (attachment, relative) in loose: return
        target = output/attachment/relative if namespace else output/relative
        data = read_data()
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'conflicting {stage} restore: {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+'.tmp')
        temporary.write_bytes(data); temporary.replace(target)
    for path in paths:
        attachment = origin(path)
        if stage in path.parts:
            save('/'.join(path.parts[path.parts.index(stage):]), path.read_bytes, attachment)
        elif path.name.endswith(('.tar.gz', '.tar', '.tgz')):
            with tarfile.open(path) as archive:
                for member in archive:
                    if member.isfile() and stage in Path(member.name).parts:
                        save(member.name, lambda: archive.extractfile(member).read(), attachment, True)
        elif path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if not name.endswith('/') and stage in Path(name).parts:
                        save(name, lambda: archive.read(name), attachment, True)
