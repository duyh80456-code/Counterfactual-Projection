"""Discover extracted or archived Kaggle Phase-0 recipe checkpoints."""
import os
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

import torch


def _walk(root):
    seen = set()
    for directory, dirs, files in os.walk(root, followlinks=True):
        real = Path(directory).resolve()
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        yield Path(directory), dirs, sorted(files)


def _recipes(root):
    matches = {}
    for directory, _, files in _walk(root):
        paths = [directory / name for name in files
                 if name.startswith('epoch_') and name.endswith('.pt')]
        epochs = []
        valid = False
        for path in paths:
            try:
                epochs.append(int(path.stem.split('_')[-1]))
            except ValueError:
                pass
            if not valid:
                try:
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                    valid = isinstance(checkpoint, dict) and checkpoint.get('kind') == 'vna_recipe'
                except Exception:
                    pass
        if valid:
            matches[directory.resolve()] = sorted(set(epochs))
    return matches


def _safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe Phase-0 archive member: {name}')
    return path


def _recipe_member(name):
    path = PurePosixPath(name)
    return path.parent.name == 'recipe' and path.match('epoch_*.pt')


def discover_phase0(input_root='/kaggle/input', output_root='/kaggle/working/phase0'):
    """Prefer extracted recipes; unpack matching archives without links or traversal."""
    input_root, output_root = Path(input_root), Path(output_root)
    matches = _recipes(input_root)
    if not matches:
        archives = [directory / name for directory, _, files in _walk(input_root)
                    for name in files if name.endswith(('.tar.gz', '.tgz', '.zip'))]
        for index, path in enumerate(archives):
            destination = output_root / f'archive_{index:03d}'
            if path.suffix == '.zip':
                with zipfile.ZipFile(path) as archive:
                    members = archive.infolist()
                    if not any(_recipe_member(m.filename) for m in members):
                        continue
                    for member in members:
                        _safe_name(member.filename)
                        if (member.external_attr >> 16) & 0o170000 == 0o120000:
                            raise ValueError(f'Unsafe Phase-0 archive link: {member.filename}')
                    for member in members:
                        if member.is_dir(): continue
                        target = destination.joinpath(*_safe_name(member.filename).parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(member))
            else:
                with tarfile.open(path) as archive:
                    members = archive.getmembers()
                    if not any(_recipe_member(m.name) for m in members):
                        continue
                    for member in members:
                        _safe_name(member.name)
                        if not (member.isfile() or member.isdir()):
                            raise ValueError(f'Unsafe Phase-0 archive member type: {member.name}')
                    for member in members:
                        if not member.isfile(): continue
                        target = destination.joinpath(*_safe_name(member.name).parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as source:
                            target.write_bytes(source.read())
        matches = _recipes(output_root)
    if not matches:
        print(f'Phase-0 checkpoints not found; {input_root} tree (depth 4):', flush=True)
        for directory, dirs, files in _walk(input_root):
            depth = len(directory.relative_to(input_root).parts)
            if depth >= 4: dirs[:] = []
            if depth > 4: continue
            print(f'{directory}/', flush=True)
            if depth < 4:
                for name in files:
                    path = directory / name
                    print(f'  {path} ({path.stat().st_size} bytes)', flush=True)
        raise RuntimeError(f'No vna_recipe Phase-0 checkpoints under {input_root}')
    chosen = min(matches, key=lambda p: (not set(range(20, 201, 10)).issubset(matches[p]), str(p)))
    if len(matches) > 1:
        print(f'Found {len(matches)} Phase-0 directories; prefer epochs 20..200: {chosen}', flush=True)
    print('Phase-0 directory:', chosen, flush=True)
    print('Phase-0 epochs:', matches[chosen], flush=True)
    return chosen
