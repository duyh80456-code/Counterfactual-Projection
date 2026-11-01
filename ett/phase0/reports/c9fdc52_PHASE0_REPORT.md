# ETT Phase 0 — completed, gate not passed (2026-11-01)

Source: `c9fdc5244d0a37a954c9afa7a00174c87456050c`. Both registered CPU runs completed their full sequences: SCR1,000,000 and PM1,200,000 updates. Each setting has20 anchors × h1/10/100 (60 cases,120 total). Native data/init/LR and all scientific settings were unchanged.

**Gate: NOT PASSED. Stop after Phase0; no Phase1/2/3 is launched.**

Gate uses immediate net, a one-sided paired sign test with exact-zero ties omitted, and median(LCBP−LC)>0 for h10 or100. Secondary net_post is never used to pass the gate. Checkpoints are paired along one seed’s trajectory; this is the registered diagnostic, not independent-seed benchmark evidence.

## Primary and secondary results

| Setting | h | median GE | median ΔC (future) | median net | IQR net | positive/20 | sign p | median CBP−C | median GM−C | median net_post | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| scr | 1 | -0.0001269728 | 0 | -0.0001269728 | 0.004747264 | 9/20 | 0.6761971 | 0.000109911 | 0 | -7.811189e-05 | N/A (h1) |
| scr | 10 | 0.0001685917 | 0 | 0.0001685917 | 0.005668133 | 10/20 | 0.5880985 | 0.0003517121 | 0 | -0.0004062057 | FAIL |
| scr | 100 | -0.001217887 | 0 | -0.001217887 | 0.01354396 | 9/20 | 0.7482777 | -0.002573907 | 0 | 0.001470447 | FAIL |
| pmnist | 1 | -8.285046e-06 | 0 | -3.868341e-05 | 0.001205064 | 4/20 | 0.9977875 | 0.000304997 | 0.001965165 | -0.008204296 | N/A (h1) |
| pmnist | 10 | -0.006185204 | 0 | -0.007990256 | 0.01584399 | 4/20 | 0.9987116 | 0.0002589524 | 0.01372209 | -0.01189038 | FAIL |
| pmnist | 100 | -0.01002941 | 0 | -0.01007944 | 0.03237291 | 5/20 | 0.994091 | -0.006813109 | 0.02219442 | -0.01333287 | FAIL |

SCR h10 has a positive median but fails the sign-test threshold. SCR h100 and both PM gate horizons have negative medians. A positive SCR h100 net_post does not reverse the pre-registered immediate-net result.

## Buffer and first-order diagnostic

All recorded cycles used1000 trained samples,800 old samples for G/pool and200 newer trained samples for acceptance diagnostics. Features were recomputed at the final expanded model. Evaluation used1000 future samples; the post diagnostic trained every branch100 steps and evaluated on the next disjoint1000 samples. Phase0 records raw contraction jumps; tau acceptance/fallback is not applied, so a fallback rate would be misleading.

| Setting | finite h1 ratios | median GE_train/(η||g_b||²) | Q25 | Q75 |
|---|---:|---:|---:|---:|
| scr | 19/20 | 0.821447 | 0.7552122 | 0.9740428 |
| pmnist | 19/20 | 0.6622669 | 0.4159402 | 0.8809246 |

Ratios use b0 loss after one SGD step on each branch and g_b computed on b0 after expansion, not future evaluation loss or buffer-ranking energy. One case/setting has zero gradient energy and an undefined ratio. These finite-LR diagnostics are not an eta→0 experiment or a general proof.

## Per-layer contraction distributions

Each entry below is median [Q25,Q75], pooled over that layer’s60 cases. g_actual_data is a summed squared preactivation residual on the800 G samples; ΔC_check is a mean-loss difference on200 disjoint trained samples. Their units differ. Dead fraction is the logged eligible-removal dead count divided by expanded width; protected real units are excluded from that count.

| Setting/site | g_actual_data | ΔC_check train | ΔC_check eval | eligible dead fraction | virtual kept fraction |
|---|---:|---:|---:|---:|---:|
| scr/0 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0.5 [0.5, 0.6666667] | 1 [1, 1] |
| pmnist/0 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0.09090909 [0.09090909, 0.1818182] | 1 [1, 1] |
| pmnist/2 | 0.01174433 [0, 13.59891] | 0 [0, 7.987022e-05] | 0 [0, 7.987022e-05] | 0 [0, 0.1818182] | 1 [1, 1] |
| pmnist/4 | 0 [0, 0.9511856] | 0 [0, 8.866191e-07] | 0 [0, 8.866191e-07] | 0 [0, 0.2727273] | 1 [0, 1] |

## Provenance, runtime and anomalies

- scr: 740.52s wall, 1 verified native anchors reused, 0 failed numeric cycles, 0 cycles with Cholesky retry. GPU usage:0.
- pmnist: 1355.52s wall, 5 verified native anchors reused, 0 failed numeric cycles, 0 cycles with Cholesky retry. GPU usage:0.
- Reuse verified old checkpoint file/state_dict hashes and the actual dataset fingerprint; loaded model hashes were checked again. All branch measurements are from the new source/tag. Old81a7d04 branch outputs remain INVALID_WINDOW and were not used.
- Cycle ID logs contain inclusive [first,last] plus SHA256 of the full compact ASCII JSON list. No full ID list is serialized in cycle logs.
- Startup and periodic storage thresholds are512 MiB. Obsolete task-created VNA test runtime/cache were cleaned after checking they were unused; no native checkpoints or result files were deleted.
- Runtime is Phase0 diagnostic cost with unequal native reuse across settings, not a full ETT-vs-BP overhead estimate. No <=3% overhead or compute-matched accuracy claim is made.
- Only T11 was run for this change:1 passed,21 deselected in5.56s. No extra tests, full pytest or smoke.
- Phase3 BN prerequisite is documented only: Gram and GradMax pool must be evaluated in blocks matching native train batch size. Not implemented or benchmarked here.

Artifacts: phase0_report.json, summary.csv, contraction_distributions.csv, raw JSONL/CSV/logs, plots, phase0_results.zip.
