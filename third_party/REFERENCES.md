# Research references

- `One-Shot-TAS-CCIL`: https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git
- inspected branch: `ccil-residual-capacity`
- inspected commit: `ec3ebc4ca7e468835935848016233204e0025b4a`

The checkout is intentionally ignored. Clone it into
`third_party/One-Shot-TAS-CCIL` when the optional adapter or audit trail is
needed.

## Official comparison sources

The four-arm Kaggle notebook clones comparison implementations at fixed
revisions instead of vendoring or rewriting them:

- RepAn: `https://github.com/xfey/RepAn.git` at
  `7cb05e93cdcd8cd83f18da1b5514bad75bc68e16`. That revision has no license
  file, so the run manifest records its license as unspecified.
- ExpandNets: `https://github.com/GUOShuxuan/expandnets.git` at
  `065d4d3aebfeb442c02227d1d5c16ee11a518945` (BSD-3-Clause terms in a file
  whose heading says "MIT License"; the manifest preserves this ambiguity).
- RepOptimizers: `https://github.com/DingXiaoH/RepOptimizers.git` at
  `2e45ff5388e9d7aabf112d7e2973df8183e6c6d9` (MIT).

Only data/budget/checkpoint/metric wrappers live in this repository. External
operators are imported from those checkouts at runtime.
