"""Build the four-process CPU regression performance pilot."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.build_kaggle_pilot_common import notebook


def build(commit): return notebook(commit, b5=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    p.add_argument('--output', default=str(ROOT/'notebooks/kaggle_vna_pilot_b5.ipynb'))
    args = p.parse_args()
    Path(args.output).write_text(json.dumps(build(args.commit), indent=1)+'\n')

if __name__=='__main__': main()
