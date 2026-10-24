# Inc10 / P2 / CertSwap seed-0 pilot

This is a separate four-arm pilot. Existing notebooks and P1 outputs remain
historical records. The new namespace is `outputs/vna_v5_inc10_p2/`.

The local handoff uses the existing immutable base commit
`3584b04d2ce14b6d8723eaf17a2800917a53b960` plus a mandatory source ZIP. The
base pin alone does not contain this uncommitted implementation. After all
source edits are integrated, generate both the source snapshot and notebook:

```bash
.venv/bin/python -m scripts.build_kaggle_vna5_inc10_p2 \
  --commit 3584b04d2ce14b6d8723eaf17a2800917a53b960 \
  --source-bundle outputs/vna5_inc10_p2_handoff/vna_v3_source.zip
```

Attach that exact `vna_v3_source.zip` to Kaggle. Its SHA256 is embedded in
`SOURCE_DIGEST`; the notebook verifies the ZIP before installing the local
source overlay, and the run identity records both base commit and source
digest.
No commit or push is performed by this handoff.

Kaggle inputs: CIFAR-100 containing `cifar-100-python/{train,test,meta}`; only
this pilot's prior output archive for resume; and the exact source ZIP matching SOURCE_DIGEST.
Enable Internet, T4×2, and the existing `github_token` secret. No P1 or clipping
study result/checkpoint is needed or reused.

GPU0: fresh Inc10/P2 authority preflight, then VNA+CertSwap seed 0 with
`--solver gram --curvature power10 --gir-rule certswap --increment 10
--warmup-epochs 1`. GPU1 starts baseline seed 0. CBP and ReDo seed 0 enter the
shared queue; either free GPU can continue them. All four use clipping off,
noBN LR scale 0.5, superclass order, native 100/50 epochs and memory 2000.
No seeds 1–2 and no additional competitors are authorized by this notebook.

Candidate clarification: preserve the existing native architecture (SkipInit
already supplies its zero-initialized convolution biases). CertSwap uses exactly
the incoming weight AND bias already selected by tiny_V. It applies neither
median recentering nor another sign choice. Every candidate, accepted or
rejected, logs its activation fraction across sample and spatial positions
using the cached block input.

`theory/s9d_dormant.tex` and `theory/check_t32.py` are not present in this
checkout; the implementation follows the attached T32 algorithm and the final
candidate clarification. No claim is made that the external theory checks
have been reproduced.

Task-2/task-10 reporting is observational and does not gate full training.
`P2_PILOT_REPORT.json` retains metrics, per-application VNA/CertSwap timing,
CertSwap rejection reasons and per-site task diagnostic fractions. Preflight
prints and archives its actual K0/timing data; no real-data timing is available
at this code-only handoff. No tests or training are run during this handoff.
