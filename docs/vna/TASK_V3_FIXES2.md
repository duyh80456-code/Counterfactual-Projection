# v3 fixes, round 2 (supervisor, 2026-10-18)

Branch `vna-v3-pilot`, HEAD 3f9ee67. Follow AGENTS.md. Do not commit. Env: `.venv/bin/python`,
`REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

1. **Exact-check time.** In `V3Engine._apply` (methods/vna_v3.py), time the exact CG check separately as
   `exact_check_seconds` (logged in the row, 0 when not run) and exclude it from `seconds`-based timing used by
   the K_base schedule: calibration points (`adapters/vna_v3_hooks.py` Calibration.apply, and
   `finalize_calibration` in experiments/pilot_v3.py which uses the selected μ trial's `seconds`) must use
   `seconds - exact_check_seconds`. When `apply=False` (the μ trials in `preflight`), do not run the exact check.
2. **CBP GIR/GradMax fallback.** In `LoPV3.install` (`generate`): if `len(pilot.hook.buffer) < 32`, keep CBP's
   native initialization for that replacement (no prepare/install) and append an event
   `dict(kind='cbp_replace', ..., gir_fallback=True, swaps=0)`. Applies to every CBP arm with GIR or GradMax.
3. **kernel_self selection.** kernel_self does not use A: skip `_select` and the alternating-A step; `active` =
   all valid sites (no `no_positive_directions` skip caused by empty A).

Tests: one short targeted test per item (put each next to the related existing test). Run only those three new
tests. Do not run the full suite. Report diff summary.
