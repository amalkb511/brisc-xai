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
| `revision/E1_per_image_<tag>.csv`, `E1_selection_<tag>.csv`, `E1_controls_<tag>.csv` | `E1_external_pituitary.py` | External check on 1,762 pituitary-adenoma slices (123 patients) of an independent dataset, 20 frozen models, no retraining (Supplementary S7): per-image measurements, slice selection and instrument controls |
| `revision/E2_summary_<tag>.csv`, `E2_by_plane_<tag>.csv`, `E2_correct_vs_wrong_<tag>.csv` | `E2_external_summary.py` | External check next to the BRISC pituitary values: means with 95% hierarchical-bootstrap intervals, by imaging plane, correct vs wrong predictions |
| `revision/E3_control_brisc_tf_float32_*.csv`, `E3_control_brisc_tf_mixed_float16_*.csv` | `E3_external_diagnostics.py control` | Control with the training loader (`tf.image.resize`): the 20 models reproduce the accuracies of Table 3 and a pituitary sensitivity of 0.986-0.996; float32 and mixed_float16 give the same values (`_summary` and `_per_run`) |
| `revision/E3_control_brisc_summary.csv`, `E3_control_brisc_per_run.csv` | `E3_external_diagnostics.py control` | Same control with the `load_img` loader of `03_measure_xai.py` (nearest neighbour): mean accuracy 0.06 to 2.1 points below Table 3 (largest for the BatchNorm architectures), which motivated E4 |
| `revision/E3_variants_*.csv`, `E3_intensity_stats.csv` | `E3_external_diagnostics.py variants` | Diagnostics of the low external sensitivity: fixed transformations of the external slices and intensity statistics |
| `revision/E4_training_loader_comparison.csv` | `E4_loader_sensitivity.py --loader training` | Sensitivity of the saliency measurements to the image loader: training loader (bilinear) against the loader of `03_measure_xai.py` (nearest neighbour); published table is `data_tables/xai_per_image.csv` |
| `revision/E4_keras_loader_comparison.csv` | `E4_loader_sensitivity.py --loader keras` | Reproducibility check: the script run with the loader of `03_measure_xai.py` unchanged against the published table |
| `gradcam_2x2_summary.csv` | `05_gradcam_target_2x2.py` | Target x precision check, all 20 models (Table 7) |
| `gradcam_2x2_summary_19models.csv` | `05_gradcam_target_2x2.py` (summary recomputed from `../data_tables/gradcam_2x2.csv` without CNN_GAP seed 1337) | Target x precision check on 19 models (Table 8; float16/float32 factors 2.75, 2.76, 7.55) |
| `figure_data/` | `B7_fig45_v2.py`, `B7_redraw_fig4_fig5.py`, `B7_fig2_v3.py` | Values plotted in Figures 2, 4, 5 |

Tags of the E1/E2 files (`<tag>`): `primary` = raw external intensities, primary selection (lesion >= 0.2% of the frame); `sens50` = same with lesion >= 0.5% of the frame; `matched_primary` and `matched_sens50` = the same two selections after matching the external intensities to BRISC.

Note: `A6_null_convergence.py` estimates the same quantity as the null in `03_measure_xai.py` (each map scored against the mask of another image of the same class, averaged over R repeats) but samples differently: `03` draws a within-class permutation per repeat, so every mask is used exactly once, whereas `A6` draws the partner mask independently and uniformly among the other images of the class (without a fixed point). The two have the same expectation but different sampling variability, so their R = 5 values differ slightly (e.g. 0.0833 vs 0.0841 for Proposed/glioma, below one standard deviation).
