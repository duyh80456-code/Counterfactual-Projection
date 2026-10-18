# Competitor survey for GIR + VNA + KKT-certified KD constraint (fixed-size CIL on PyCIL iCaRL)

Survey date: 2026-10-19. Setup: CIFAR-100 B0 Inc5 (20 tasks), superclass class order, ResNet-32 (+/- BN), memory 2000.

Legend: **[V]** means the fact was checked against a search result or fetched page (source given). **UNVERIFIED** means it was not confirmed this session. "PyCIL" means listed in the LAMDA-CL/PyCIL README "Methods Reproduced" section. The repo was fetched at github.com/LAMDA-CL/PyCIL; G-U-N/PyCIL redirects there.

---

## 0. PyCIL model list (fetched README) [V]

FineTune, EWC, LwF, Replay, **GEM**, **iCaRL**, **BiC**, **WA**, **PODNet**, **DER**, PASS, RMM, IL2A, ACIL, SSRE, FeTrIL, Coil, **FOSTER**, **MEMO**, **BEEF**, DS-AL, SimpleCIL, Aper, **TagFex** (CVPR 2025, added 2025-04).
- Backbones named in the README: ResNet32 (CIFAR-100) and ResNet18 (ImageNet).
- **Not in PyCIL:** LUCIR, DER++, A-GEM, GPM, and every plasticity method.
- LUCIR exists in LibContinual (huggingface.co/RL-MIND/LibContinual core/model/lucir.py) and in Mammoth (`lucir`). [V]
- No 2024-2026 fixed-size, exemplar-based method is in mainline PyCIL. TagFex expands the network (see E). The newest fixed-size, exemplar-based methods built on PyCIL are external repos: MRFA, C-Flat, Dual-Arch and Forget-Forgetting (see A and D).

---

## A. Methods that explicitly target both loss of plasticity and forgetting (fixed size)

| Method | Venue [V] | Link | Code | CIL with exemplars? | Notes |
|---|---|---|---|---|---|
| **UPGD** (Elsayed & Mahmood), "Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning" | ICLR 2024 [V: proceedings.iclr.cc 2024 PDF; arXiv 2404.00781 / 2302.03281] | proceedings.iclr.cc/paper_files/paper/2024/file/8e5f0591943d8dae5702af12dcdcd2f6-Paper-Conference.pdf | github.com/mohmdelsayed/upgd [V] | **No.** Streaming, one pass, no replay. Benchmarks: input-permuted MNIST, label-permuted EMNIST/CIFAR-10/mini-ImageNet, PPO MuJoCo. | The canonical "both" method. Forgetting is measured on label-permuted tasks. S&P is a baseline there and "holds performance at a lower level". Must be ported to PyCIL as an optimizer swap. |
| **Weight Space Consolidation** (Cho, Moon, Chunara, Cho, Cha), "Forget Forgetting: Continual Learning in a World of Abundant Memory" | ICLR 2026 [V: proceedings.iclr.cc 2026 PDF; arXiv 2502.07274] | proceedings.iclr.cc/paper_files/paper/2026/file/17fc58b1ba92abbb72c5f457e140d851-Paper-Conference.pdf | "supplementary material". No public repo found: **UNVERIFIED** | **Yes, run in PyCIL.** CIFAR-100 and ImageNet-100, 10 tasks, 20-500 exemplars/class, 5 seeds. | Closest precedent to ours: rank-based partial reset (plasticity) plus SWA-style weight averaging (stability), fixed size. Ablation: "Only when both are combined do we observe substantial gains." Its Table 5 compares Random, Revert, S&P and CBP resets by accuracy only. **Strongest direct competitor.** |
| **FIRE** (Han et al.), "Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff" | ICLR 2026 [V: iclr.cc/virtual/2026/poster/10010833; proceedings hash 8a47999f...] arXiv 2602.08040 [V] | arxiv.org/abs/2602.08040 | Project page isaac7778.github.io/fire/ [V]; GitHub repo **UNVERIFIED** | Partly. Has a class-incremental vision setting (20 phases) but no replay buffer described. Stability is measured as weight distance (SFE), **not forgetting**. | Explicitly "stability-plasticity". Baselines: S&P, DASH, Parseval, L2-Init, CBP, SNR, ReDo, Muon, full reset. Supplies the baseline suite and hyperparameters. |
| **Dual-Arch** (Lu et al.), "Rethinking the Stability-Plasticity Trade-off in CL from an Architectural Perspective" | ICML 2025, PMLR v267 (lu25t) [V] | arxiv.org/abs/2506.03951 | github.com/byyx666/Dual-Arch [V] | **Yes.** Plug-in on iCaRL, WA, DER, FOSTER, MEMO with PyCIL hyperparameters and augmentation. CIFAR-100 and ImageNet-100, 10 and 20 tasks. | Not strictly fixed size: two smaller networks (plastic + stable), about 33-52% fewer total params; only the stable net is used at inference. Reports AAN (plasticity) and FAF (forgetting). Good plug-in competitor on the same plane. |
| **C-Flat** (Bian et al.), "Make Continual Learning Stronger via C-Flat" | NeurIPS 2024 [V: proceedings.neurips.cc 2024] | arxiv.org/abs/2404.00986 | github.com/WanNaa/C-Flat [V] | **Yes.** Plug-in on Replay, iCaRL, WA, PODNet, DER, FOSTER, MEMO. CIFAR-100 **B0_Inc5**, ResNet-32, seed 1993: the same protocol as ours. Also run on GPM-family methods. | Fixed-size optimizer plug-in (flatness). Reports old/new accuracy for iCaRL, PODNet, FOSTER at B0_Inc5 (Table 7). Single runs. Not framed as loss of plasticity, but it is the nearest plug-in-on-PyCIL comparator. |
| **ANCL** (Kim et al.), auxiliary-network CL | CVPR 2023 [V: openaccess.thecvf.com] | arxiv.org/abs/2303.09483 | github.com/kim-sanghwan/ANCL [V] | Exemplar-based CIL variants **UNVERIFIED** (framework base also UNVERIFIED) | Trains an auxiliary plastic network and interpolates regularizers. Fixed size at inference. Optional. |
| Linear Connector (Lin et al.), "Towards Better Plasticity-Stability Trade-off in IL" | CVPR 2022 [V] | openaccess.thecvf.com (CVPR 2022) | github.com/lingl1024/Connector [V] | No. Data-free, Adam-NSCL based, per-task heads (task-IL-like). | Weight averaging of an NSCL model and a plastic model. Optional or related-work only. |
| NBSP (Lan et al.), "Neuron-level Balance between Stability and Plasticity in DRL" | arXiv 2504.08000 [V]; venue **UNVERIFIED** | arxiv.org/abs/2504.08000 | none found | RL only | Skill-neuron gradient masking plus replay. Related work only. |
| Neuroplastic Expansion (Liu et al.) | ICLR 2025 [V] | proceedings.iclr.cc 2025 e094d1e3... | github.com/torressliu/neuroplastic-expansion [V] | RL only, **grows** the network | Dormant pruning, growth, and replay. Related work only. |
| Reset & Distill / "Prevalence of Negative Transfer in Continual RL" (Ahn et al.) | ICLR 2025 [V: proceedings hash ba9e3d60...] | arxiv.org/abs/2403.05066 | UNVERIFIED | RL only | Reset plus distillation to fight forgetting. Related work only. |

No paper from 2023-2026 was found that targets **dormant-unit reset in exemplar-based CIL with KD and a certified old-output constraint**. Searches covered dormant-neuron CIL, reset-with-replay and plasticity injection in CIL. The closest are Forget-Forgetting (PyCIL, reset plus averaging), C-Flat and Dual-Arch (PyCIL plug-ins) and UPGD (streaming).

---

## B. Plasticity-only methods

| Method | Venue [V] | Code | Reports forgetting? | Setting |
|---|---|---|---|---|
| **CBP**, Dohare et al., "Loss of plasticity in deep continual learning" | Nature 632 (2024) [V: nature.com s41586-024-07711-7; ideas.repec] | github.com/shibhansh/loss-of-plasticity [V: arXiv 2306.13812 footnote] | **No.** In class-incremental CIFAR-100 the network trains on all data of all seen classes "to focus on plasticity rather than on forgetting" [V] | Class-incremental CIFAR-100: 5 classes, then +5 for 20 increments, 200 epochs each, ResNet-18. Baselines: base, S&P, head reset, retrain-from-scratch. |
| **ReDo**, Sokar et al., "The Dormant Neuron Phenomenon in DRL" | ICML 2023, PMLR v202 [V] | github.com/google/dopamine/tree/master/dopamine/labs/redo [V: arXiv 2302.12902 footnote] | No (RL) | Reinits incoming weights and zeros outgoing weights (approximately function-preserving). The most direct analogue of GIR's re-install step. |
| **Shrink & Perturb**, Ash & Adams, "On Warm-Starting Neural Network Training" | NeurIPS 2020 [V: proceedings.nips.cc 2020] | official repo UNVERIFIED (simple to implement) | No | Warm-start, stationary data |
| **L2-Init / regenerative regularization**, Kumar, Marklund, Van Roy | **CoLLAs 2024** (3rd CoLLAs), published in PMLR v274 (2025), pp. 410-430 [V: proceedings.mlr.press/v274/kumar25a; mlanthology collas/2024] | none listed | No | Permuted / label-shift streams |
| **SWR**, Hernandez-Garcia, Dohare, Luo, Sutton, "Reinitializing weights vs units for maintaining plasticity" | **4th CoLLAs, held 2025** ("Published at 4th CoLLAs, 2025" on arXiv 2508.00212); PMLR **v330**, pp. 776-806 [V]. The brief's "CoLLAs 2026" label refers to the PMLR publication year; the conference is CoLLAs 2025. | github.com/JFernando4/collas_2025_swr_paper [V] | **No.** "Forgetting was not considered in this paper"; old classes are kept in training [V] | Permuted MNIST; class-incremental CIFAR-100 with a ViT (5+5, 20 tasks, 100 epochs/task) |
| **SNR**, "Self-Normalized Resets for Plasticity in CL" | ICLR 2025 [V: proceedings.iclr.cc 2025 hash 359c0c4d...] | github.com/ajozefiak/SelfNormalizedResets [V] | No (calls LoP "distinct from" forgetting) | Permuted MNIST-style streams, Permuted Shakespeare |
| **Plasticity Injection**, Nikishin et al. | NeurIPS 2023 [V: proceedings.neurips.cc 2023] | UNVERIFIED | No (RL, Atari) | Trainable-param count is unchanged [V, abstract]. A frozen copy is kept, so total params grow: from method knowledge, **UNVERIFIED** here. Arguably not fixed size; mention only. |
| **DASH**, Shin, Oh, Cho, Yun | NeurIPS 2024 [V] | none found (UNVERIFIED) | No | Stationary warm-start (noise memorization). The arXiv page fetch was rate-limited. |
| **Parseval regularization**, Chung et al. | NeurIPS 2024 [V] | none listed | No; authors "focus mainly on improving plasticity rather than stability" [V] | RL only (Gridworld, CARL, MetaWorld) |
| Hare & Tortoise, Lee et al. | ICML 2024, PMLR v235 (lee24d) [V] | github.com/dojeon-ai/hare-tortoise [V] | No standard forgetting metric | Warm-start, 10-chunk CL (not class-split), Atari |
| Spectral regularization, Lewandowski et al. | ICLR 2025 [V] | none listed | No (trainability focus) | Includes class-incremental CIFAR/Tiny-ImageNet; baselines L2, L2-Init, S&P, ReDo, CReLU, Wasserstein reg., streaming EWC |
| AdaLin, Rohani et al. | arXiv 2505.09486, venue UNVERIFIED | none | No | Class-split CIFAR-100, 5 classes/task |

**Evidence that plasticity methods increase forgetting when combined with replay or KD CIL.** No paper found reports a forgetting metric for CBP, ReDo, S&P, L2-Init, SNR or SWR in exemplar-plus-KD CIL. This gap is our opening. Indirect evidence only:
1. **Forget Forgetting (ICLR 2026, PyCIL):** reset alone is insufficient. At 20 exemplars/class, Random reset scores 45.30 vs Replay 48.63. Resetting every iteration is worse than resetting once. Gains need reset plus averaging. [V]
2. **FIRE (ICLR 2026):** full reset and DASH show "a sharp drop immediately after each reset". DASH has the highest weight drift (SFE), which "may contribute to an erasure of useful learned knowledge". S&P avoids sharp drops but is suboptimal. [V]
3. **Joudaki et al., ICLR 2026** ("Barriers for Learning in an Evolving World"): dropout worsens forgetting in bit-flipping. [V]
4. **Query-Only Attention (arXiv 2510.00365, preprint):** CBP shows severe forgetting in Slowly-Changing-Regression backward tests, like vanilla BP. [V, but not peer reviewed]
5. **UPGD (ICLR 2024):** on label-permuted (forgetting) tasks, S&P plateaus at a lower level. [V]
6. **CBP Nature and SWR:** forgetting was deliberately excluded from measurement. [V]

Paper framing: "Prior plasticity methods were evaluated without forgetting measurement (Dohare 2024; Hernandez-Garcia 2025). We are the first to measure them on the stability-plasticity plane in exemplar-plus-KD CIL." This claim is consistent with everything found. Phrase it as "to our knowledge".

---

## C. Stability-side constrained / projection methods

| Method | Venue [V] | Code | Plasticity cost (documented) |
|---|---|---|---|
| **GEM** | NIPS 2017 | **in PyCIL** [V] | QP per step; same inequality-constraint family as our KKT constraint. Natural ablation anchor. |
| **A-GEM** | ICLR 2019 [V: iclr.cc/virtual/2019/poster/715] | github.com/facebookresearch/agem [V]; Mammoth `agem`, `agem_r` [V] | Averaged constraint, weaker than GEM. |
| **GPM** | ICLR 2021 (oral) [V: iclr.cc/virtual/2021/oral/3429] | github.com/sahagobinda/GPM [V: ecosyste.ms] | Fully orthogonal updates mean "poor plasticity" as tasks accumulate (Zhao et al., CVPR 2023 [V]). TRGP: projection "can throttle the learning performance for the new task" [V]. |
| **Adam-NSCL** | CVPR 2021 [V] | UNVERIFIED (expected github.com/ShipengWang/Adam-NSCL) | Null-space restriction; the Connector paper (CVPR 2022) exists to fix its plasticity. |
| **TRGP** | ICLR 2022 spotlight [V] | Code was in supplementary [V]; gitee.com/adolfk/TRGP mirror [V, search]; LibContinual has core/model/trgp.py [V] | Built to relax GPM's plasticity loss. |
| Rethinking GP / stability-plasticity feature-space decoupling (Zhao et al.) | CVPR 2023 [V] | none on CVF page | Names the GPM plasticity problem explicitly. |
| FS-DGPM, SGP, ROGP | UNVERIFIED venues (C-Flat applies to La-/FS-DGPM [V]; ROGP arXiv 2301.12131 seen) | — | optional |

Caveat: GPM, TRGP and Adam-NSCL are mostly evaluated in task-incremental, multi-head settings. Our setting has a single head with exemplars, so they are an awkward fit. Note: **GPM's repo includes "20-task CIFAR-100 Superclass"** [V], so our superclass split has a direct protocol precedent there (see F).

---

## D. Fixed-size exemplar CIL methods

| Method | Venue | In PyCIL? | Code |
|---|---|---|---|
| iCaRL (our base) | CVPR 2017 | **yes** | PyCIL |
| **PODNet** | ECCV 2020 [V] | **yes** | github.com/arthurdouillard/incremental_learning.pytorch [V] |
| **WA** | CVPR 2020 [V: arXiv 1911.07053] | **yes** | PyCIL |
| **BiC** | CVPR 2019 [V] | **yes** | PyCIL; LibContinual |
| LUCIR | CVPR 2019 [V] | **no** | github.com/hshustc/CVPR19_Incremental_Learning [V]; Mammoth `lucir`, LibContinual [V] |
| DER++ | NeurIPS 2020 [V] | **no** | Mammoth `derpp` [V] (github.com/aimagelab/mammoth) |
| GEM | NIPS 2017 | **yes** | PyCIL |
| Coil | ACM MM 2021 | yes | PyCIL |
| **MRFA** (Zheng, Zhou, Ye, Zhan) | ICML 2024, PMLR v235 (zheng24p) [V] | fork only (pycil-mrfa-added docs) | github.com/bwnzheng/MRFA_ICML2024 [V, docs]. PyCIL base implied by the PyCIL-fork docs; UNVERIFIED in the paper. Fixed-size plug-in on rehearsal. |
| **C-Flat** | NeurIPS 2024 [V] | external, PyCIL-style | github.com/WanNaa/C-Flat [V] |
| **Dual-Arch** | ICML 2025 [V] | external, PyCIL settings | github.com/byyx666/Dual-Arch [V] |
| Mammoth plug-ins (CaSpeR, CSCCT, LiDER on iCaRL, DER++, ER-ACE) | venues UNVERIFIED | no (Mammoth) | Mammoth model keys `icarl_casper`, `icarl_cscct`, `icarl_lider` [V] |

Kim et al. (CVPR 2023, "On the Stability-Plasticity Dilemma of CIL") [V] found that most CIL methods (LUCIR, PODNet, AFC, ...) "heavily favor stability over plasticity": their feature extractors barely improve after the base stage, while DER improves. The paper uses feature accuracy (linear probe) and CKA. **Adopt their feature-accuracy metric as a plasticity measure**; it directly supports our motivation.

---

## E. Growth/expansion methods (upper-bound table)

| Method | Venue [V] | PyCIL? | Code |
|---|---|---|---|
| **DER** | CVPR 2021 (oral) [V] | **yes** | github.com/Rhyssiyan/DER-ClassIL.pytorch [V] |
| **FOSTER** | ECCV 2022 | **yes** | github.com/G-U-N/ECCV22-FOSTER [V, docs] |
| **MEMO** | ICLR 2023 | **yes** | github.com/wangkiw/ICLR23-MEMO [V, docs] |
| BEEF | ICLR 2023 | yes | github.com/G-U-N/ICLR23-BEEF [V, docs] |
| **TagFex** | CVPR 2025 [V] | **yes** (added 2025-04) | github.com/bwnzheng/TagFex [V]. Expansion on DER: 61.6M params on CIFAR-100 10-10, same as DER. |
| **GradMax** | ICLR 2022 [V] | no | github.com/google-research/growneuron [V]. Zeroes **incoming** weights and sets outgoing weights by SVD to maximize gradient norm. No CL experiments. |
| **TINY** (Verbockhaven et al.) | **TMLR 2024** [V: HAL record "Transactions on Machine Learning Research ... 2024"]; arXiv 2405.19816 | no | gitlab.inria.fr/mverbock/tinypub [V]. Small-amplitude γ init, not exactly function-preserving. No CL experiments. |

GradMax and TINY have no CL code, so they are better as **ablations of GIR's init** (GIR-init vs GradMax-init vs TINY-init vs ReDo-init inside our pipeline) than as table rows.

---

## F. Precedent for superclass-ordered CIFAR-100 plus loss-of-plasticity measurement

- **CIFAR-100 Superclass, 20 tasks x 5 classes, one superclass per task:** introduced in APD (Yoon et al., ICLR 2020 [V: arXiv 1902.09432; iclr.cc/virtual/2020/poster/1441]). Used in GPM's official repo [V]. Those works do not state whether task IDs are given at test time; the GPM line is task-incremental in practice. Our B0 Inc5 superclass order is **the same task partition in a class-incremental, single-head setting.**
- Ramasesh et al., "Anatomy of Catastrophic Forgetting" (arXiv 2007.07400 [V]; ICLR 2021 per search snippet, otherwise UNVERIFIED): superclass-based distribution-shift tasks. Shows semantic similarity drives forgetting.
- The continuum library offers a superclass (`labels_type="category"`) class-incremental scenario [V], but at the coarse-label level.
- **The LoP literature uses 5+5 class-incremental CIFAR-100 with 20 tasks** (Dohare Nature 2024 [V]; SWR [V]; AdaLin [V]), but with *random* order and *no forgetting* (all old data retained).
- **No existing protocol combines a superclass order, exemplar-limited CIL and LoP measurement.** Searches covered superclass, LoP and CIL combinations. Say "to our knowledge" and cite APD/GPM for the split and Dohare/SWR for the 5x20 LoP protocol.

---

## G. Recommended competitor list

### Must-run
1. **iCaRL (PyCIL)**: the base. Every result is a delta over it.
2. **PODNet (PyCIL)**: the strongest stability-biased fixed-size KD method; Kim et al. 2023 show it trades away plasticity. Tests our Pareto claim against it.
3. **WA (PyCIL)**: cheap, strong fixed-size baseline that is also a C-Flat and Dual-Arch host, so numbers cross-check against published tables.
4. **GEM (PyCIL)**: the same inequality-constraint family as our KKT constraint. Isolates the value of "certified in KD metric plus active repair" over gradient-space QP.
5. **iCaRL + ReDo**: the closest analogue to GIR (dormant reinit, zero outgoing weights). Official code. Shows dormant-unit reinit alone vs growth-informed reinit.
6. **iCaRL + CBP**: the Nature-paper standard. Official code. Never evaluated for forgetting, so measuring it fills that gap.
7. **iCaRL + Shrink & Perturb**: a simple universal LoP baseline used in every LoP paper. Indirect evidence of reset-induced forgetting.
8. **iCaRL + UPGD**: the canonical "both LoP and forgetting" method. Official code. A fair test of the joint claim, run as an optimizer swap.
9. **iCaRL + C-Flat**: a fixed-size plug-in with official code, already run on PyCIL iCaRL at CIFAR-100 B0_Inc5 / ResNet-32. The most protocol-compatible plug-in competitor.

### Should-run
10. **iCaRL + L2-Init**: the regularization-side LoP baseline (CoLLAs 2024). Trivial to implement.
11. **Forget-Forgetting (reset + weight averaging)**: the only PyCIL reset-plus-stability method (ICLR 2026). The most similar idea. Reimplement from the paper if its code is not public.
12. **iCaRL + FIRE**: the newest stability-plasticity reinit (ICLR 2026). Its paper supplies the S&P/DASH/SNR/Parseval hyperparameter grid.
13. **iCaRL + SNR** or **SWR**: recent reset methods with official code. Pick one; SWR suits the without-BN/LN variant, where its authors report it beats unit resets.
14. **Dual-Arch on iCaRL**: an ICML 2025 stability-plasticity plug-in with official code, reported with plasticity and forgetting metrics. Note its different parameter budget.
15. **MRFA on iCaRL**: a 2024 fixed-size rehearsal plug-in from the PyCIL authors. A recent-SOTA fixed-size row.
16. **BiC (PyCIL)**: a bias-correction baseline that completes the standard fixed-size table.
17. **DER, FOSTER, MEMO (PyCIL)**: the growth upper-bound table. Free to run and directly relevant to our "virtual expansion" claim.

### Optional
18. **LUCIR / DER++** via LibContinual or Mammoth: classic baselines outside PyCIL, so a fair port takes effort.
19. **A-GEM, GPM, TRGP**: projection-family baselines with code, but task-IL oriented. Useful for a projection-cost discussion or a small task-IL side table.
20. **TagFex (PyCIL)**: a 2025 expansion SOTA for the upper-bound table.
21. **GradMax, TINY, ReDo as init ablations inside GIR**: tests "growth-informed" against alternative function-preserving inits rather than as standalone competitors.
22. **DASH, Parseval, Hare & Tortoise, spectral regularization, ANCL**: completeness only. Mostly stationary-data or RL methods, and few have code.
23. **Plasticity Injection**: skip or mention only. It is RL-only, and keeping a frozen network copy likely breaks the fixed-size constraint (UNVERIFIED).

Metrics to report for all methods: final and average incremental accuracy; average forgetting; new-task accuracy (plasticity); Kim et al. 2023 feature accuracy (linear probe); dormant-unit fraction. Use 3+ seeds and the same exemplar herding.

### Unread or blocked pages
- DASH arXiv HTML (2410.23495): rate-limited (HTTP 429).
- OpenReview PDF "The Dual Nature of Plasticity Loss in Deep Continual RL": fetch error.
- HAL TMLR record for TINY: blocked by a bot check. The venue comes from the search snippet only.
