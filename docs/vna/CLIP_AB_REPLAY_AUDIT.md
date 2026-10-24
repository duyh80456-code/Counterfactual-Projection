# Audit of the first strict clipping A/B replay

Input: `vna4_clip_AB.tar.gz`, SHA256
`7d436eefc1b7da04e6dae7be4b44bf51f00875c40fa351579db5b70538ff6348`.
Replay source: `c42995dd101466d66ca9809a00a986841b7d6322`.
Baseline s0 reference: `d88abc496eb1fe21d1b6df66d90b47cf93d7ded1`.

## B s0: zero observed loss mismatches

The archive contains104 completed epoch losses: task1 epochs1–100 and task2
epochs1–4. All104 match the original Python float values exactly, with zero
absolute/relative tolerance. The remaining946 full-horizon epoch losses are
unavailable, beginning at task2 epoch5. They are missing observations, not
numerically different losses.

| Task | Epoch | Original train loss | Replayed train loss | Result |
|---|---|---|---|---|
| 1 | 100 | 0.4936316817998886 | 0.4936316817998886 | Exact match |
| 2 | 1 | 3.9622376230027943 | 3.9622376230027943 | Exact match |
| 2 | 2 | 3.6820567581388683 | 3.6820567581388683 | Exact match |
| 2 | 3 | 3.6069644225968256 | 3.6069644225968256 | Exact match |
| 2 | 4 | 3.5040721363491483 | 3.5040721363491483 | Exact match |
| 2 | 5 | 3.3845521608988443 | Unavailable | Epoch interrupted |
| 2 | 6 | 3.3123777442508273 | Unavailable | Not run |

B s0 was `STOPPED_REPLAY_MISMATCH`, not a numerical divergence or an observed
loss mismatch. Its norm history records2165 training updates:2000 in task1,
4*36 in the four completed task2 epochs, and21/36 updates in task2 epoch5.
The full epoch5 average cannot be reconstructed from its partial loss.

The old `loss_check.status=FAIL` represented failure to cover all1050 epochs.
Its946 `differences` all had `current=null`; there were no differences between
two observed numeric losses. The report now separates `mismatched_epochs`,
`missing_current_epochs`, `missing_reference_epochs` and `matched_epochs`.
An otherwise matching interrupted B replay is `INCOMPLETE`, never a PASS or
a certificate for the full20-task run.

The full epoch-by-epoch CSV and machine-readable audit were generated under
`outputs/clip_ab_audit/` from this unchanged archive.

## Why the shared stop marker was set

Baseline s2 control reproduced loss failure at globalstep2008. Its six clipped
variants all completed60 updates. Full s2 ran the native correction at2001,
with K=K0=512, s=3.9223728179673483 and zero swaps. The intervention check passed.

Full s2 then reached step2010 with finite loss2667261184.0 and a nonfinite
float32 gradient norm. The new monitor raised `tensor=gradient` before SGD;
the registered original producer only stopped for nonfinite loss, at2011.
The mismatch is between stopping criteria; these records alone do not prove
that the replay's trajectory differed from the original. The newly added
pre-gradient and post-parameter guards violated measuring-only behavior for
None controls.

Fix: with tau=None keep measuring norms and observing updates, but do not add
new stopping criteria. Main retains its original loss/epoch checks. Competitors
retain their original upstream/method checks (including the existing native
CBP/ReDo/S&P/UPGD post-update parameter guard). With a non-None threshold the
new clipping safeguards remain enabled. The exact2008/2011/task-step6 control
gates, frozen K0, VNA event requirement, selected tau and late-divergence policy
remain unchanged. A new T4 replay must confirm2011; it is not asserted from
CPU tests.

## Local validation of the repair

Affected-file CPU tests:139 passed (main, competitor wrappers, clipping,
study and collection); a further115-test focused run passed after checking the
native loss-guard caller and report formatting. Tests cover finite-gradient
norm overflow and Inf gradients in None controls, the original next-loss
failure boundary, retained guards with clipping enabled, CBP/GEM shared runner
behavior, and missing-versus-mismatched epoch reports. No T4 rerun is claimed.
