# ETT decisions — preregistered 2026-11-01

Authority: the self-contained ETT prompt supplied 2026-11-01. Historical VNA
code/results remain archived; new ETT imports no VNA/GGN/CG/Jacobian stack.
Only atomic artifact helpers and upstream data/CBP code are reused.

## Staging
Implement core + T1–T10 + real Phase0 first. Do not implement/launch the later
benchmark phases before reporting Phase0 and receiving the next instruction.
No old H1–H4 notebook is repurposed to imply it runs ETT. No full pytest or smoke.

## Fixed Phase0 protocol before observing data
- SCR: upstream `lop/slowly_changing_regression/cfg/sgd/bp/relu.json` architecture,
  target generator, 20 inputs/5 hidden ReLU/1 output, 1e6 single-sample SGD steps,
  changing15 bits/10000 steps, target100, beta0.7. Use the first registered LR0.01.
- PM-small: 3×10 ReLU, single-example SGD0.01, raw MNIST /255, no normalization,
  20 complete permutations of the 60000 training examples. The prompt does not
  specify task count; choose20 before results. This is not claimed to reproduce
  SWR's original protocol; no SWR baseline or paper result is claimed at Phase0.
- Seeds data1101/init2201/pool3301 are independent and shared across paired arms.
  Checkpoint pool seed is3301+native step. One experimental seed only.
- rho0.1, horizons1/10/100;20 eligible anchors after trained-buffer warmup,
  including the first and final task tails. See Revision1 for reserve/evaluation
  sizes and validated native-checkpoint reuse. No tau gate in Phase0.
- Statistics use the Revision1 rolling trained-sample split. No extra SGD
  updates for ETT apart from the separately declared post-measurement branch.
  Compute cost includes candidate backward, post-training forwards and algebra.
- Upstream GnT utility is observed without resets on the BP trajectory; at each
  branch checkpoint call its actual test/reset once (replacement rate0.1,
  maturity100, adaptable_contribution, decay0.99). Its stochastic no-replacement
  cases are logged, not silently forced to replace a unit. GradMax-reset shares
  the selected slots, cadence and reset draws; only incoming initialization changes.

## Numerical interpretation and unresolved mathematical ambiguities
- G_reg is the prescribed feature-ridge metric, not the raw data Gram. OBS costs
  and compensation are exact in that supplied metric (AppendixB). Report both
  `g_selected_metric` and `g_actual_data`; do not claim ridge costs equal raw LS
  residuals. T5–T7 compare unregularized full-rank problems and AppendixB.
- An all-zero feature block has zero prescribed ridge scale. Use the C1 exact-dead
  path before any factorization. No arbitrary damping floor is added. For genuine
  remaining failures after the prescribed Cholesky retries, log and stop with failure
  rather than fabricate a result or silently retain a wider deployed model. The
  native Phase0 checkpoint remains resumable. This is deliberately stricter than
  “skip the layer”: skipping cannot leave an expanded deployment, and no fallback
  LS solution exists without changing the numerical rule. Report such a failure.
- Without an outgoing bias but with a following BN, omission of the fitted
  intercept b changes preactivation by -b. To preserve the fitted eval function,
  running_mean must change by -b (C3a); this corrects the prompt's plus sign.
  There is no real outgoing bias to copy. Train-mode shift invariance still holds.
- Exact-dead preservation applies only to exactly zero features. The prescribed
  near-dead threshold is a numerical candidate rule, not an exact C1 identity.
- Adam clocks are coordinate-shaped. This implements row/column clocks even when
  both axes can have different birth times. All reset arms must use the same
  optimizer in any future Adam setting. Phase0 uses SGD only.
- Slots follow Lemma0; optimizer moments follow retained columns despite LS
  changing their outgoing values, as explicitly requested. Unrelated parameter
  objects remain unchanged. No reset/warmup is added after transplant.
- Unknown residual branches and LayerNorm/GroupNorm sites are ineligible. Conv
  adapters accept only single-consumer 3×3, ungrouped zero-padding conv pairs or
  basic-block conv1→bn1→relu→conv2. Stem/residual output sites are excluded.

## Gates and costs
One-sided paired sign test drops exact ties. A complete setting/horizon needs20
anchors, median(net)>0, p<0.05 and median(LCBP-LC)>0 for h10 or100. Also report
IQR, positive fractions, component losses, pool prediction and virtual retention.
Passing this diagnostic is not the final benchmark success criterion. Stop after
Phase0 either way; Phase1/2 are not automatically authorized by a positive gate.
Phase0 branch wall times are not a full ETT-vs-BP overhead estimate. No claim of
<=3% overhead or compute matching until the full-run phase measures it.

Configuration files use the JSON subset of YAML1.2, parsed by stdlib json.
This avoids adding PyYAML to the established .venv CPU runtime.

First real Phase0 attempt at362e8a6 failed before the first valid measurement:
rebuild installed float64 LS weights into a float32 model. Preserve parameter
dtype/device during remap; do not change data, seeds or hyperparameters. Retain
the failed logs and use a new source-tagged output for the corrected run.

At81a7d04 SCR hit a zero feature Gram at anchor52625. Core incorrectly tried
Cholesky before the mandatory C1 dead-first branch. Move exact-zero elimination
before factorization; no added ridge floor or changed selection heuristic.
T2 now also checks this singular, all-dead case with BN train/eval adapters.


## Revision1 — authoritative window correction (2026-11-01)
Supersedes every earlier window/evaluation statement above. All measurement
outputs from81a7d04 are INVALID_WINDOW, including the historical partial report.
Keep the raw files and native BP checkpoints; never use those losses for a gate.

F1: SCR and PM-small each use1000 *trained* samples, not batches. Samples enter
a CPU input/label ring after every completed update, including outside a cycle.
The old chronological800 form G and the GradMax pool; the newest200 are the
acceptance check. Recompute features at fixed final expanded weights; never
store feature vectors or aggregate features across changing weights. Contract
retains the ring. Restore the ring exactly for general ETT checkpoint resume.

MLP capacity=max(1000,20*sum(p_layer)); SCR p7 and PM p36 both yield1000.
Conv-only capacity=max(5,max_layer ceil(20*p_layer/rows_per_image)), p=9(n+k)+1.
The adapter caller must declare the actual unfold rows per image for each site;
missing geometry is rejected. Mixed conv/linear chains also satisfy the MLP
floor. Conv geometry is a Phase2 prerequisite, not guessed or benchmarked here.

A cycle waits for a full trained buffer. Phase0 first eligible anchor is1000,
not0; the random untrained checkpoint has no historical samples and cannot meet
F1/F4. Keep valid saved81a7d04 anchors if >=1000 and the current-task reserve fits,
then fill the registered20 anchors evenly. Native trajectories/seeds/LR/data are
unchanged. Only native checkpoints are reused: compare native protocol, dataset
fingerprint, checkpoint file SHA256, saved state_dict SHA256 and loaded weights.
Store the hashes and old source in the new protocol. Rerun every branch under
new source/tag; do not import any old branch metric or old selected coefficient.
Reconstruct pre-anchor input history from the deterministic stream indices,
so anchor checkpoints contain model, optimizer and GnT history, not duplicate
input buffers. The full latest checkpoint still records RNG/measurement progress.

Primary evaluation:1000 untrained current-task examples at anchor+100..1099.
After that measurement, train N/C/CBP/GM each100 further ordinary stream steps,
starting at anchor+h, with paired RNG and data. Secondary evaluation uses the
next disjoint1000 examples at anchor+1100..2099. Shift anchors earlier to keep
all2100 reserved samples in the current task. net_post is explicitly secondary;
the original immediate-net sign/median/CBP gate is unchanged.

At h1 compute g_b on b0 *after expansion with the selected incoming weights*.
Use GE_train=LN(b0 after one step)-LE(b0 after one step) for its C4b ratio.
Pool ranking energy on the older800 buffer samples is logged separately.
GradMax-reset uses the same G-buffer pool as ETT. Eval/train acceptance losses
are computed on the newest200 trained samples, never a next-stream batch.

Storage preflight requires2 GiB free before data creation or experiment output.
Recheck during checkpoint cadence and pause with state when space drops below
that threshold. No new Phase0 measurements are authorized by a failed preflight.
Task-created compressed downloads/SCR data cache may be regenerated from fixed
seeds/checksums; no checkpoints or historical outputs are deleted to free space.

Targeted verification: only T11/T12 and T2/T9 affected by core buffer/signature
changes. core/expand.py is unchanged, so no T1 rerun; no full test or smoke.
