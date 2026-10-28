"""Code provenance is separate from strict experiment identity."""
from pathlib import Path
import subprocess


def code_commit():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
        text=True).strip()


def source_hashes():
    """Hash executable project sources so a dirty checkout cannot masquerade as its pin."""
    import hashlib
    root=Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ('methods','projection','experiments','adapters')
        for path in sorted((root/folder).rglob('*.py'))}
