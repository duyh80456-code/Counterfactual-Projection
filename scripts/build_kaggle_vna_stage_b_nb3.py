"""Build pinned Notebook 3: GPU0 B1 resnet8x4/wrn_16_2; GPU1 gated B2 WRN."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.build_kaggle_vna_stage_b_rest import build as build_rest


def build(commit): return build_rest(commit, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    p.add_argument('--output', default=str(ROOT/'notebooks/kaggle_vna_stage_b_nb3.ipynb'))
    args = p.parse_args(); path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(args.commit), indent=1)+'\n')


if __name__ == '__main__': main()
