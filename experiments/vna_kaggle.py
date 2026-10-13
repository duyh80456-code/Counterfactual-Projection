"""VNA output restore/archive helpers using the established Kaggle discovery."""
from pathlib import Path
import json
import os
import shutil
import tarfile
import tempfile
import zipfile

from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.shared_protocol import atomic_json_save, atomic_torch_save, validate_resume_protocol


def restore_outputs(input_root, output, protocol):
    output = Path(output)
    recipe = output / "recipe"
    recipe.mkdir(parents=True, exist_ok=True)
    cache = output / "_resume_cache"
    def compatible(payload):
        validate_resume_protocol(payload.get("protocol"), protocol)
        return True
    try:
        target = recipe / "latest.pt"
        if target.exists():
            import torch
            validate_resume_protocol(torch.load(target, map_location="cpu", weights_only=False).get("protocol"), protocol)
        found, rejected = discover_checkpoints(input_root, cache, kind="vna_recipe",
            payload_filter=compatible,
            payload_transform=lambda p: {"epoch": p["epoch"], "full": "optimizer" in p})
        full = [item for item in found if item["payload"]["full"]]
        if full:
            latest = max(full, key=lambda item: item["payload"]["epoch"])
            target = recipe / "latest.pt"
            if target.exists():
                import torch
                if torch.load(target, map_location="cpu", weights_only=False)["epoch"] > latest["payload"]["epoch"]:
                    latest = None
            if latest:
                temporary = target.with_suffix(".pt.tmp")
                shutil.copy2(latest["path"], temporary)
                temporary.replace(target)
        for item in found:
            epoch = item["payload"]["epoch"]
            if epoch % 10:
                continue
            import torch
            payload = torch.load(item["path"], map_location="cpu", weights_only=False)
            atomic_torch_save({k: payload[k] for k in ("kind", "epoch", "model", "protocol")},
                              recipe / f"epoch_{epoch:03d}.pt")
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    def marker(name, stream):
        path = Path(name)
        if path.name.startswith("epoch_") and path.suffix == ".json" and "completed" in path.parts:
            payload = json.load(stream)
            if not {"identity", "model_sha256", "epoch", "rows"}.issubset(payload):
                raise ValueError("invalid diagnostic marker")
            target = output / "signal" / "completed" / path.name
            if target.exists() and json.loads(target.read_text()) != payload:
                raise ValueError(f"conflicting diagnostic output: {name}")
            atomic_json_save(payload, target)
    for directory, _, files in os.walk(input_root, followlinks=True):
        for filename in files:
            path = Path(directory)/filename
            if filename.endswith(".json"):
                with path.open() as stream:
                    marker(str(path), stream)
            elif filename.endswith((".zip", ".tar", ".tgz", ".tar.gz")):
                if zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path) as archive:
                        for name in archive.namelist():
                            if name.endswith(".json") and "completed" in Path(name).parts:
                                with archive.open(name) as stream:
                                    marker(name, stream)
                else:
                    with tarfile.open(path, "r|*") as archive:
                        for member in archive:
                            if member.isfile() and member.name.endswith(".json") and "completed" in Path(member.name).parts:
                                with archive.extractfile(member) as stream:
                                    marker(member.name, stream)
    print("Resume checkpoints:", len(found), "excluded:", len(rejected))


def archive_outputs(output, budget=5_000_000_000):
    """Hard-link atomic immutable outputs before tar creation, even during training."""
    output = Path(output)
    target = output.with_suffix(".tar.gz")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="vna_snapshot_") as temporary:
        snapshot = Path(temporary)
        for folder in ("recipe", "signal"):
            for path in (output/folder).rglob("*"):
                if path.is_file() and not path.name.endswith(".tmp") and not any(part.endswith(".lock") for part in path.parts):
                    dest = snapshot/path.relative_to(output)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    os.link(path, dest)
        for log in output.glob("*.log"):
            shutil.copy2(log, snapshot/log.name)
        raw = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())
        # During atomic replacement old archive + temporary archive coexist.
        if raw * 3 > budget:
            raise RuntimeError(f"Output/archive peak estimate {raw*3} exceeds {budget}")
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
