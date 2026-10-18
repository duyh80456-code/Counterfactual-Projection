"""Code provenance is separate from strict experiment identity."""
from pathlib import Path
import subprocess


def code_commit():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
        text=True).strip()
