# Pilot v3 historical results

**Thuật toán trước T26, không dùng để so với v4.**

The requested three-seed v3 B5 and O1 `er`/`er_vna` Kaggle outputs are not present in this checkout. No values or commits are inferred. Existing local pilot summaries below are earlier two-task synthetic wiring runs, not those completed Kaggle experiments. Raw local records were backed up to `/private/tmp/vna_cleanup3_raw_logs.tar.gz` before cleanup; external Kaggle runs and archives are untouched.

| Source | Arm | Seed | Commit | Tasks | Metric | Value |
|---|---|---|---|---|---|---|
| `b3_icarl_gir_randdir_summary.json` | icarl_gir_randdir | 1993 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | last task cnn_top1 | 6.88 |
| `b3_icarl_gir_summary.json` | icarl_gir | 1993 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | last task cnn_top1 | 5.62 |
| `b3_icarl_gir_vna_summary.json` | icarl_gir_vna | 1993 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | last task cnn_top1 | 5.0 |
| `b3_icarl_ngd_summary.json` | icarl_ngd | 1993 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | last task cnn_top1 | 5.0 |
| `b3_icarl_summary.json` | icarl | 1993 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | last task cnn_top1 | 5.0 |
| `b3_icarl_vna_summary.json` | icarl_vna | 1993 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | last task cnn_top1 | 5.0 |
| `b5_bp_ngd_l0.1_summary.json` | bp_ngd_l0.1 | 0 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | mean_last50 | 0.018254489462826217 |
| `b5_bp_ngd_l1_summary.json` | bp_ngd_l1 | 0 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | mean_last50 | 0.0182247481278002 |
| `b5_bp_summary.json` | bp | 0 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | mean_last50 | 0.018257804550768045 |
| `b5_bp_vna_l0.1_summary.json` | bp_vna_l0.1 | 0 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | mean_last50 | 0.01825506853530042 |
| `b5_bp_vna_l1_summary.json` | bp_vna_l1 | 0 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | mean_last50 | 0.018230468418227435 |
| `b5_cbp_gir_randdir_summary.json` | cbp_gir_randdir | 0 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | mean_last50 | 0.01823923025781939 |
| `b5_cbp_gir_summary.json` | cbp_gir | 0 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | mean_last50 | 0.009383577075953782 |
| `b5_cbp_gir_vna_l1_summary.json` | cbp_gir_vna_l1 | 0 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | mean_last50 | 0.009383577075953782 |
| `b5_cbp_summary.json` | cbp | 0 | a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8 | 2 | mean_last50 | 0.018246396858268597 |

| Requested v3 run | Seeds | Metric / commit |
|---|---|---|
| B5 (all completed arms) | 0, 1, 2 | unavailable locally |
| O1 er / er_vna | as run on Kaggle | unavailable locally |
