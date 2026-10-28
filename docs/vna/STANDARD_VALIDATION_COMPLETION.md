# Remaining validation work — 2026-10-29

GPU checkpoint verification was explicitly skipped by the user. The local host
has no CUDA checkpoint evidence. No GPU result, real-data selected lambda or
accuracy/forgetting improvement is claimed, and no extra Input is required.

H3-small now schedules the first full-window correction in each task and then
every200 updates:3 opportunities/task at2,202,402,300 over100 tasks. The old
4000 schedule remains an ablation. This addresses coverage, not empirical
cadence optimality. Corrections can cost much more than native training; exact
T4 runtime is unmeasured and multiple11-hour sessions may be needed. Resume and
registered task/epoch horizons remain intact.

H2 automatically calibrates lambda over{0,0.1,1,10,100} on held-out validation
from seen experiences, starting task2 and bounded to3 windows. Trials share the
same checkpoint/window and fixed validation tensors. Every trial restores
weights, buffers, optimizer, gradients and RNG. Minimum validation CE among
certified nonzero accepted candidates selects the weight (ties:smaller lambda).
The decision is frozen, checkpointed and restored after interruption. If all
windows fail qualification, FAILED_PREFLIGHT prevents an unqualified full run.
Neither nonincrease gates nor solver certificates are relaxed. Validation
calibration is a one-step local proxy, not full-run tuning or proof that lambda1
was bad. It uses external benchmark experience metadata; native online updates
remain task-ID-free. Costs and chosen values are recorded separately from test
accuracy; main and numerical ablation policies have the same selection budget.

Native LPR is unchanged. H4's scope remains training-only prior-domain loss
constraints; no learned domain-specific representation mechanism is claimed.

The dedicated temporary Python3.10 environment uses pinned requirements-h2,
torch2.2.2/vision0.17.2 and Avalanche0.4/typing4.4. Tests run against actual
Avalanche without the Python3.12 Protocol shim. H2_AVALANCHE_FINAL_ENVIRONMENT.json
records installed versions, official source checks and the sole allowed torch
versus typing metadata exception. No permanent global environment was modified.

Logs linked here are CPU synthetic/API integration evidence only. The tuned
Avalanche fixture uses native/unconstrained correction to exercise validation
selection and state restoration; a separate fixed-weight H2 fixture retains
protected tiny_V. No synthetic selected weight is copied into benchmark configs.

Final CPU evidence:
- STANDARD_VALIDATION_AVALANCHE_TESTS.log:29 passed,0 skipped in82.71s in the
  actual Python3.10/Avalanche0.4 environment (19 expected no-logger warnings).
  Includes fixed-weight protected tiny_V and validation-tuned native/unconstrained
  fixtures, native LPR parity, split/feature APIs, replay/geometry gates, native
  ImageNet arm/resume wiring, cadence simulation and checkpoint transactions.
- STANDARD_VALIDATION_NATIVE_LPR_TESTS.log:1 passed in6.92s, explicitly verifies
  that enabling the main config's calibration policy does not calibrate or
  change native LPR, including its bitwise native-oracle comparison and resume.
- STANDARD_VALIDATION_LOCAL_TESTS.log:50 passed in200.47s, affected correction,
  state/rollback, ResNet32 BN, H3/H4 actual-engine and resume fixtures in .venv.
  These groups overlap; they are not independent benchmark experiments.

Validation is restricted to classes already observed by online updates, not
future labels from a partially encountered experience. Seen-class state,
selection attempts, selected weight and RNG are restored under interruption.
The protocol records the calibration rule; the selected value is in summary,
geometry_selection.json and comparison.csv. No benchmark lambda is fabricated.

Reproduce from repository root (Python3.10 environment separate from .venv):

```bash
python3.10 -m venv /tmp/vna_h2_final_env
/tmp/vna_h2_final_env/bin/python -m pip install torch==2.2.2 torchvision==0.17.2 -r requirements-h2.txt -r requirements-upstream.txt pytest==8.3.5 exceptiongroup==1.2.2
/tmp/vna_h2_final_env/bin/python -m pip install --no-deps avalanche-lib==0.4.0 typing-extensions==4.4.0
REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg /tmp/vna_h2_final_env/bin/python -m pytest tests/test_standard_validation_phase.py tests/test_standard_benchmark_protocols.py tests/test_standard_finalize.py tests/test_standard_compare_notebooks.py -q -rs -o addopts=''
```

No full repository pytest or real-data benchmark training was run. GPU checkpoint
verification remains intentionally skipped, not passed. More H3 opportunities
and H2 calibration are prospective experimental changes, not retrospective proof
that the previous cadence or weight caused a failure.
