"""VNA output restore/archive helpers using the established Kaggle discovery."""
from pathlib import Path
import os
import shutil
import tarfile
import tempfile
import time
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
                    for attempt in range(3):
                        try:
                            os.link(path, dest)
                            break
                        except FileNotFoundError:
                            if attempt == 2:
                                raise
                            time.sleep(.01*(attempt+1))
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


def restore_stage_archives(input_root, output, stage, *, exclude_parts=(), include_names=()):
    """Expanded files win over their own older archive; other conflicts refuse."""
    input_root, output = Path(input_root), Path(output)
    paths, visited = [], set()
    for directory, dirs, files in os.walk(input_root, followlinks=True):
        resolved = Path(directory).resolve()
        if resolved in visited:
            dirs[:] = []
            continue
        visited.add(resolved)
        dirs[:] = [name for name in dirs if name != '.git']
        paths.extend(Path(directory)/name for name in files if (Path(directory)/name).is_file())
    paths.sort()
    def attachment(path):
        parts=path.relative_to(input_root).parts
        return '/'.join(parts[:3] if parts[0] in ('notebooks','datasets') else parts[:1])
    def relative(name):
        parts=Path(name).parts
        if '..' in parts: raise ValueError('unsafe archive member')
        if stage not in parts:return None
        result=Path(*parts[parts.index(stage)+1:])
        if (not result.parts or result.name.endswith('.tmp') or set(result.parts).intersection(exclude_parts)
                or (include_names and result.name not in include_names)):return None
        return result
    loose = {}
    for path in paths:
        name=relative(path)
        if name is not None:loose[(attachment(path),name)]=path
    restored=[]
    def save(name, read_data, source, archived=False):
        if archived and Path(name).is_absolute():raise ValueError('unsafe archive member')
        name=relative(name)
        if name is None or (archived and (source,name) in loose):return
        data=read_data()
        target = output/name
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'conflicting {stage} restore: {target}; incoming source={source}:{name}')
        restored.append(str(target))
        if target.exists():return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+'.tmp')
        temporary.write_bytes(data); temporary.replace(target)
    for path in paths:
        source=attachment(path)
        if stage in path.parts:
            save('/'.join(path.parts[path.parts.index(stage):]), path.read_bytes, source)
        elif path.name.endswith(('.tar.gz', '.tar', '.tgz')):
            with tarfile.open(path) as archive:
                for member in archive:
                    if (member.isfile() and stage in Path(member.name).parts and
                        not set(Path(member.name).parts[Path(member.name).parts.index(stage)+1:]).intersection(exclude_parts)):
                        save(member.name, lambda:archive.extractfile(member).read(), source, True)
        elif path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if (not name.endswith('/') and stage in Path(name).parts and
                        not set(Path(name).parts[Path(name).parts.index(stage)+1:]).intersection(exclude_parts)):
                        save(name, lambda:archive.read(name), source, True)
    return restored
