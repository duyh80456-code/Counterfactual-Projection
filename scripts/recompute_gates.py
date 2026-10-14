"""Select the official Stage-B gates from saved per-mu results, without measurements."""
import argparse
import json
from pathlib import Path


def recompute_gates(stage_a_summary, root):
    official_mu = float(json.loads(Path(stage_a_summary).read_text())['mu_rel_default'])
    if official_mu not in (1e-2, 1e-3):
        raise ValueError('Stage A mu_rel_default must be 1e-2 or 1e-3')
    root = Path(root)
    merged_path = root / 'gate.json'
    merged = json.loads(merged_path.read_text()) if merged_path.exists() else {}
    updates = []
    for folder in sorted(root.iterdir()):
        gate_path = folder / 'gate.json'
        if not folder.is_dir() or not gate_path.exists():
            continue
        gate = json.loads(gate_path.read_text())
        by_mu = {str(mu): json.loads((folder / f'gate_mu_{mu}.json').read_text())
                 for mu in (1e-2, 1e-3)}
        gate.update(by_mu=by_mu, official_mu=official_mu,
                    status=by_mu[str(official_mu)]['status'])
        updates.append((gate_path, gate))
        merged[folder.name] = gate
    for path, gate in updates:
        path.write_text(json.dumps(gate, indent=2) + '\n')
    merged_path.write_text(json.dumps(merged, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-a-summary', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('outputs/stageB'))
    args = parser.parse_args()
    recompute_gates(args.stage_a_summary, args.root)


if __name__ == '__main__':
    main()
