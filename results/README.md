# Result tables

Tables produced by the scripts and used in the manuscript and Supplementary Material. All were produced
under protocol hash `6f82d66818c0`.

| File | Produced by | Used for |
|---|---|---|
| `revision/ablation_runs_main20.csv` | `02_train_ablation.py` | Test accuracy per architecture x seed (20 runs; Table 3) |
| `revision/ablation_runs_A4_SE_only.csv` | `A4_train_se_only.py` | CNN_GAP_SE control (reviewer A4) |
| `revision/A2_neardup.csv` | `A2_near_duplicate_audit.py` | Near-duplicate audit (pHash) |
| `revision/A5_*` | `A5_degenerate_diagnosis.py` | Degenerate Grad-CAM map diagnosis |
| `revision/A6_null_convergence_fast.csv`, `A6_fast_log.txt` | `A6_null_convergence.py` | Permuted-null convergence |
| `revision/A7_threshold.csv`, `A7_log.txt` | `A7_threshold_sensitivity.py` | Dice threshold sensitivity (Table S3) |
| `revision/A8_hier_36_contrasts.txt` | `A8_hierarchical_stats.py` | Hierarchical bootstrap contrasts (Table S2) |
| `revision/B3_*` | `B3_boot_v1.py` (reproduces `B3_perclass_boot.csv` from `B3_input_pred.csv`), `B3_perclass_confusion.py` | Confusion matrices, per-class metrics (Table 6; Tables S4-S5) |
| `revision/B4*` | `B4_pituitary_gap.py`, `B4b_saliency_location.py`, `B4c_peak_tests_v2.py`/`v3.py`, `B4d_argmax_control_v1.py` | Pituitary / saliency location analyses |
| `revision/A10_efficiency.csv`, `A10_env.txt` | `A10_efficiency.py` (float32, seed-42 weights) | Parameters, FLOPs, latency, memory |
| `gradcam_2x2_summary.csv` | `05_gradcam_target_2x2.py` | Target x precision check, all 20 models (Table 7) |
| `gradcam_2x2_summary_19models.csv` | `05_gradcam_target_2x2.py` (summary recomputed from `../data_tables/gradcam_2x2.csv` without CNN_GAP seed 1337) | Target x precision check on 19 models (Table 8; float16/float32 factors 2.75, 2.76, 7.55) |
| `figure_data/` | `B7_fig45_v2.py`, `B7_redraw_fig4_fig5.py`, `B7_fig2_v3.py` | Values plotted in Figures 2, 4, 5 |

Note: `A6_null_convergence.py` estimates the same quantity as the null in `03_measure_xai.py` (each map scored against the mask of another image of the same class, averaged over R repeats) but samples differently: `03` draws a within-class permutation per repeat, so every mask is used exactly once, whereas `A6` draws the partner mask independently and uniformly among the other images of the class (without a fixed point). The two have the same expectation but different sampling variability, so their R = 5 values differ slightly (e.g. 0.0833 vs 0.0841 for Proposed/glioma, below one standard deviation).
