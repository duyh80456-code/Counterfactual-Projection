# CertSwap / Inc10 P2 verification — 2026-10-24

Final local result: **98 PASS, 0 FAIL, 5 SKIP**, 411.97 seconds.
Environment: PyTorch 2.2.2, CPU, CUDA unavailable.

| Requirement | Result |
| --- | --- |
| Per-site before uses the same suffix as swapped, without autograd for jumps | PASS |
| Actual noBN ResNet32 T32 fixture (15 blocks, 31 convolutions, over 460k parameters) | PASS |
| Dead slot: exact zero CE/KD jumps; rejected model/optimizer/RNG unchanged | PASS on CPU; CUDA pending |
| Actual `main.run(icarl_full, Inc10/P2)` dispatches to CertSwap | PASS |
| iCaRL hook and competitor loader have identical LR traces, including 1/S/S+1 at tasks 1 and 2 | PASS |
| Cached suffix vs full forward: bitwise equality across all 15 sites and after a deep edit | PASS on CPU; CUDA pending |
| Main and competitor summary identity schemas and report field names | PASS |
| Independent worktree 3584b04: default weights, optimizer, RNG and scientific logs bitwise | PASS |
| Mandatory pilot CUDA gate on a real CIFAR-100 batch | NOT RUN locally; runs first in notebook |

The historical comparison executes separate Python processes against an
unmodified detached `3584b04` worktree and the current source. It covers native
ResNet32 iCaRL training on two fixture tasks and a separate three-SGD-step plus
actual GIR application. Only timing fields and source commit/digest provenance
are excluded from scientific log comparisons.

Two synthetic-window CUDA cases and the three real-CIFAR CUDA gate cases were
SKIPPED locally because no CUDA device exists. These are not reported as PASS.
The pilot runs the three real-CIFAR tests on T4: exact dead-slot jumps,
bitwise suffix/full equality, and identical j/rho for the same candidate twice.
Any failure, error, skip or missing case blocks preflight and training.

Command:

```bash
REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python .venv/bin/python -m pytest \
  tests/test_vna_certswap.py tests/test_certswap_cuda.py \
  tests/test_p2_warmup.py tests/test_p2_pilot_identity.py \
  tests/test_p2_backward.py tests/test_p2_full_arm.py \
  tests/test_main_cil.py tests/test_competitors.py \
  tests/test_vna_v3.py tests/test_dead_schur_policy.py \
  -ra --junitxml=outputs/certswap_code_review/final_tests.xml
```

Local artifacts: `outputs/certswap_code_review/final_tests.xml` and
`outputs/certswap_code_review/final_tests.log`. No pilot preflight or training
was executed as part of this verification. No cuDNN nondeterminism has been
observed or diagnosed locally; the CUDA gate must establish the GPU result.
