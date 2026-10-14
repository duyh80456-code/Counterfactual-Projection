"""Code provenance is separate from strict experiment identity."""
from pathlib import Path
import subprocess


def code_commit():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
        text=True).strip()


def resume_code_history(checkpoint, commit):
    history = list(checkpoint.get('code_history', []))
    old = checkpoint.get('code_commit')
    if not history:
        history = [{'from_epoch': 1, 'commit': old or 'unknown'}]
    if history[-1]['commit'] != commit:
        boundary = checkpoint['epoch'] + 1
        print(f'WARNING: CODE COMMIT CHANGED {history[-1]["commit"]} -> {commit}; '
              f'epoch boundary {boundary} (after epoch {checkpoint["epoch"]})', flush=True)
        history.append({'from_epoch': boundary, 'commit': commit})
    return history
