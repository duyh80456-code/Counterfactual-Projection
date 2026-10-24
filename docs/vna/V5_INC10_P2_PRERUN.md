# Inc10 / P2 / CertSwap seed-0 pilot

This is a separate four-arm pilot. Existing notebooks and P1 outputs remain
historical records. The new namespace is `outputs/vna_v5_inc10_p2/`.

The notebook pins the complete source commit
`c2499f306c41d143db75e8135ab79d7f3f51e325` directly from GitHub.
No source ZIP is needed. Regenerate the notebook with:

```bash
.venv/bin/python -m scripts.build_kaggle_vna5_inc10_p2 \
  --commit c2499f306c41d143db75e8135ab79d7f3f51e325
```

Kaggle inputs: CIFAR-100 containing `cifar-100-python/{train,test,meta}`; only
this pilot's prior output archive for resume (optional for the first session).
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
