# ETT Phase0 revision1 — 2026-11-01

Implemented F1–F8 against current source, retaining the C1 exact-dead ordering
and the correct negative running-mean shift. Authority: ett/REVISION1.md.

- Independent rolling1000-sample trained input/label ring: older800 for Gram
  and GradMax, newer200 for acceptance. Retained across cycles and resume.
- No next-stream check argument; features recomputed at final expanded weights.
- ETT and GradMax-reset pools use the same trained G partition.
- Primary evaluation1000 future samples, plus100 post steps for N/C/CBP/GM
  and a disjoint next1000 evaluation set; net_post excluded from the gate.
- h1 C4b ratio uses native/expanded b0 loss after one SGD step, and the actual
  outgoing virtual gradient on b0 after expansion; buffer ranking energy separate.
- Native checkpoint file/state_dict hashes, native config and dataset fingerprint
  checked before reuse; loaded model hash checked again. No old branch losses
  reused. Anchor files omit duplicate buffer data.
- All old81a7d04 measurement roots and the partial report marked INVALID_WINDOW,
  files retained. Archived native anchors remain eligible subject to validation.

Only permitted targeted tests were run: T11/T12 plus T2(two variants)/T9 affected
by core signature/buffer changes. 5 passed,17 deselected in4.62s. T1 was not rerun
because core/expand.py is unchanged. See ett/tests/revision1.log.
T12 is a seeded sanity fixture, not a universal generalization guarantee.

Storage preflight is BLOCKED:163540992 bytes available versus2147483648 required
at check time. No new Phase0 data generation, branch measurement, smoke, full
pytest or Phase1/2 run was launched. The dataset cache created by this task was
partly removed to allow code edits; all checkpoints/results remain intact.

REVISION1_NATIVE_MANIFEST.json inventories hashes of two SCR and six PM native
anchor files (including unusable anchor0). Runtime excludes anchor0 because no
trained buffer exists, verifies actual stream fingerprints, records eligible
reuse hashes in protocol.json and remeasures every branch under a new source/tag.

Run only when512 MiB are free, using a fresh output directory:
.venv/bin/python -m ett.phase0.measure_ge --config ett/configs/phase0_scr.yaml \
  --out outputs/ett_phase0_revision1_<commit>/scr --data-dir /tmp/ett_data
.venv/bin/python -m ett.phase0.measure_ge --config ett/configs/phase0_pmnist.yaml \
  --out outputs/ett_phase0_revision1_<commit>/pmnist --data-dir /tmp/ett_data

Phase0 has no valid new gate result yet. Stop after Phase0.

The earlier2-GiB blocked preflight above is historical. Review follow-up lowers
the actual startup/periodic gate to512 MiB without changing measurement config.
Cycle ID logs are now inclusive bounds plus a full-list SHA256; see DECISIONS.
