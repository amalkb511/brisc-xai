# Measurement choices and reference values in Grad-CAM alignment (BRISC 2025)

Code and result tables for the manuscript *"Measurement Choices and Reference Values Shape Saliency
Alignment in Brain Tumor MRI Classification"* (JEEEMI submission #2146).

Frozen protocol hash: **`6f82d66818c0`** (`config.PROTOCOL_HASH`; stamped in every result table).

## What is here

| Path | Content |
|---|---|
| `config.py` | All paths, the frozen training protocol (`PROTOCOL`), architectures, Grad-CAM layers, thresholds. Edit paths only; changing `PROTOCOL` changes the hash. |
| `models.py` | The four architectures (Baseline, CNN_GAP, CNN_GAP_BN, Proposed = CNN_GAP_BN_SE) and the Grad-CAM wrapper. |
| `01_build_dataset.py` … `06_make_figures.py` | Main pipeline (dataset, training, Grad-CAM measurement, tables, 2x2 target/precision check, figures). |
| `A2_near_duplicate_audit.py`, `A3_equivalence_tost.py`, `A4_train_se_only.py`, `A5_degenerate_diagnosis.py`, `A6_null_convergence.py`, `A7_threshold_sensitivity.py`, `A8_hierarchical_stats.py`, `A10_efficiency.py`, `B3_*`, `B4*`, `B7_*`, `_xai_common.py` | Analyses and figure scripts added in the revision (near-duplicate audit, equivalence test, SE-only control, degenerate-map diagnosis, null convergence, threshold sensitivity, hierarchical bootstrap, efficiency, per-class metrics, pituitary analyses, figures). The header of each script states its inputs and outputs. |
| `E1_external_pituitary.py`, `E2_external_summary.py`, `E3_external_diagnostics.py`, `E4_loader_sensitivity.py` | External check on an independent pituitary-adenoma dataset (E1, E2), diagnostics of that check against the BRISC test images (E3) and sensitivity of the saliency measurements to the image loader (E4). They need the trained `.keras` models and the corresponding dataset; the header of each script states its inputs and outputs. |
| `data_tables/` | `manifest.csv` (the final train/val/test split: path, MD5 digest, perceptual-hash group; 4,243 train, 748 validation, 1,000 test) `leakage_removed.csv` (the nine training files removed because byte-identical to a test image), both as written by `01_build_dataset.py`, `gradcam_2x2.csv` (per-image target x precision check, 20 models x 860 images x 4 conditions = 68,800 rows, as written by `05_gradcam_target_2x2.py`), and `xai_per_image.csv` (per-image Grad-CAM alignment metrics, 20 models x 860 masked test images x 2 layer roles = 34,400 rows, as written by `03_measure_xai.py`). Paths in these files are those of the Colab session (`/content/...`); use `stem` and `md5` to match files. |
| `results/` | Result tables used in the manuscript (see `results/README.md`). |
| `notebook/colab_session_cleaned.ipynb` | The Colab session in which the experiments were run, outputs cleared. It is a log of what was executed, not a clean entry point. |

## Data

BRISC 2025 (Fateh et al., Scientific Data 2026, DOI 10.1038/s41597-026-06753-y, CC BY 4.0).
The dataset is **not** redistributed here. Download it from Kaggle (`briscdataset/brisc2025`) into a
clean directory (not one synchronised by Drive/Dropbox) and set `BRISC_ROOT`.

## Environment

Experiments were run on an NVIDIA L4 (23 GB) with Python 3.13 and TensorFlow 2.21.0
(`results/revision/A10_env.txt`). Training uses the `mixed_float16` policy; Grad-CAM is computed in float32.
`pip install -r requirements.txt` (lower bounds only; exact versions of the other libraries were not recorded during the runs).

## Running

```bash
export BRISC_ROOT=/path/to/brisc2025      # pristine download
export BRISC_OUT=/path/to/output          # everything is written here
python 01_build_dataset.py     # split, leakage removal, manifest
python 02_train_ablation.py    # 4 architectures x 5 seeds (resumable)
python 03_measure_xai.py       # Grad-CAM alignment metrics per image
python 04_make_tables.py
python 05_gradcam_target_2x2.py
python 06_make_figures.py
```

Revision analyses run from the repository root (they import `config`, `models`, `_xai_common`):

```bash
python -I A6_null_convergence.py --out A6_null_convergence.csv
python -I A7_threshold_sensitivity.py --out A7_threshold.csv
python A5_degenerate_diagnosis.py --out A5_per_image.csv --out-summary A5_summary.csv
python A8_hierarchical_stats.py --xai $BRISC_OUT/xai/xai_per_image.csv --out A8_hier.csv
python A10_efficiency.py --out A10_efficiency.csv          # run alone on the GPU
python A4_train_se_only.py                                  # CNN_GAP_SE control (5 seeds)
# A2, A3, B3, B4*, B7*: see the I/O contract at the top of each script
```

Neural-network training is not bit-for-bit deterministic across GPUs and library versions, so retraining may
give slightly different numbers than those reported; the result tables in `results/` are the values used in the paper.

## Smoke test

Steps 01 to 05 were run end to end from a clean copy of this repository (one seed, three epochs, NVIDIA T4, output to a separate
directory). The pipeline completed without error, the dataset step reproduced the published split and the nine
cross-partition duplicates, and the tables were written. Accuracy after three epochs is not meaningful; this test checks that
the code runs, not that the reported numbers are reproduced. `06_make_figures.py` was not part of this test.
`03_measure_xai.py` prints `RuntimeWarning: Mean of empty slice` for `cam_mean_brain_nonlesion` when a coarse map has no brain
pixel outside the lesion; that value is then NaN. The primary metric (`auc_intrabrain`) does not use this column; it is used in a
descriptive table of `04_make_tables.py` and in `B4_pituitary_gap.py`.

## Protocol integrity

`config.PROTOCOL` is hashed and the hash is written into every output row. Rows produced under different
protocols therefore cannot be pooled silently: `03_measure_xai.py` stops if the models on disk carry a different hash from `config.py`, and `04_make_tables.py` stops if the rows it reads mix several hashes.
Do not edit `PROTOCOL` to reproduce published numbers.

## Dataset note

The authors' manifest of BRISC 2025 (`sha256` column) shows seven hashes that appear in both the training and test
partitions (five pituitary, two meningioma), affecting nine training files because one test image occurs three times in
train. `01_build_dataset.py` recomputes this from per-file MD5 digests, removes the affected files from the training side only (the test set stays the
published 1,000-image benchmark) and lists them in `leakage_removed.csv`. The released metadata contains no subject
identifier, so subject-level independence between partitions cannot be established.

## Trained models

The trained weights (`.keras` files, 4 architectures x 5 seeds) are not included in this repository (see `.gitignore`);
`02_train_ablation.py` regenerates them. Parameter counts, FLOPs, latency and memory of each architecture are in
`results/revision/A10_efficiency.csv`, and the hardware and software versions in `results/revision/A10_env.txt`.

## Notes on paths

Default paths in `config.py` follow the Colab layout used for the experiments (`/content/...`). Set `BRISC_ROOT` and
`BRISC_OUT` to use other locations; no other path needs editing. Some log files under `results/revision/` still show
the Colab/Drive paths they were written to.

## Citation and licence

Code: MIT (`LICENSE`). Dataset: CC BY 4.0, see above.
Citation: to be added after publication.
