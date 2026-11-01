# Targeted ETT verification — 2026-11-01

Command: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python
-m pytest tests/test_ett_core.py -q -o addopts=''

- T1: PASS — MLP/conv, with/without BN, train/eval expansion invariance.
- T2: PASS — dead-unit contraction, MLP/conv BN train/eval.
- T3: PASS — exact removed preactivation contribution, MLP/conv.
- T4: PASS — trained virtual-slot permutation invariance, MLP/conv.
- T5: PASS — compensation versus NumPy AppendixB and torch lstsq, MLP/conv unfold.
- T6: PASS — OBS grouped costs versus direct residual and NumPy, group1/group9.
- T7: PASS — downdates match naive LS backward on20 seeded full-rank problems.
- T8: PASS — retained SGD momentum/rowstep Adam moments/clocks and BN bitwise gather.
- T9: PASS — disabled ETT versus BP, 50 SGD steps bitwise.
- T10: PASS — no training/all virtual dropped, parameters/optimizer/output bitwise.

20 passed in6.29s, no skips. Full log: T1_T10.log.
No full pytest or separate smoke was run. Phase0 is a distinct real experiment.
