# nb2/nb3: always measure both μ candidates (1e-3 and 1e-2)

No commit. Do NOT run pytest or any smoke. Allowed: compile check, rebuilding notebooks.

Stage A's rule can only return `mu_rel_default ∈ {1e-2, 1e-3}`. nb2/nb3 must not depend on nb1, so they
measure both and the gate is evaluated for both.

1. `scripts/build_kaggle_vna_stage_b_rest.py` (nb2 and nb3): every runner gets `--mu-rel-default 1e-3`
   so the measurement classes (`sorted({mu_rel_default, 1e-2})`) measure both values. Remove the
   Stage-A-summary requirement/fallback logic for choosing what to *measure*; keep the summary search
   only to report which μ is official (`OFFICIAL_MU`, `None` if no Stage A summary is attached).
2. Runners `stage_b12.py`, `stage_b3.py`, `stage_b4.py`: write `gate_mu_0.01.json` and
   `gate_mu_0.001.json` (same `evaluate_gate`, `setting_config(..., mu_rel_default=μ)`), and `gate.json`
   = `{"by_mu": {"0.01": ..., "0.001": ...}, "official_mu": <from --official-mu or null>,
   "status": <status of the official μ, or "PENDING_STAGE_A">}`. Add `--official-mu` (optional float).
   Reproduction/teacher gates unchanged.
3. Final notebook cell: table of gate status per setting for both μ, highlighting `OFFICIAL_MU` if
   known; otherwise print "official μ pending Stage A (nb1)". Merged `outputs/stageB/gate.json` keeps
   both.
4. Add `scripts/recompute_gates.py --stage-a-summary PATH --root outputs/stageB`: rewrites each
   setting's `gate.json` `official_mu`/`status` from the Stage A summary, using the existing per-μ files
   (no recomputation of measurements).

Rebuild nb2/nb3. Final message: files touched.
