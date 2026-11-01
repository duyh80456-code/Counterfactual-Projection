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
- rho0.1, horizons1/10/100;20 evenly spaced anchors including initialization and
  the final-task tail. Shift an anchor earlier if needed to keep its100 training
  and8 held-out evaluation examples in its current task. Evaluate all horizons
  on the same8 future examples starting at anchor+100. No tau gate in Phase0.
- Statistics use the last4 actual trained batches. No extra SGD updates for ETT.
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
- An all-zero feature block has zero prescribed ridge scale. No arbitrary damping
  floor is added. After the prescribed Cholesky retries, log and stop with failure
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
