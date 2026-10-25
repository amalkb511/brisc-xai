"""
Step 4 — Build every table in the manuscript from the CSVs.

Nothing here computes a model output. It reads ablation_runs.csv and
xai_per_image.csv and emits the published tables. If a number appears in the
manuscript and not in one of these outputs, it does not belong in the
manuscript.

Tables produced (in TABLES_DIR):
    t1_dataset.csv          per-class counts, duplicates, removals
    t2_ablation.csv         accuracy, params, best epoch, per architecture
    t3_mcnemar.csv          paired tests between architectures, Holm-corrected
    t4_alignment.csv        alignment per architecture x role x class
    t5_dissociation.csv     between-class tests with Cliff's delta and CI
    t6_controls.csv         instrument controls
    t7_stability.csv        is the class ordering stable across seeds?
    t8_comparison_set.csv   what the choice of comparison set costs

Run:  python 04_make_tables.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu

import config as C


def holm(p):
    p = np.asarray(p, float)
    order, m, adj, run = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return float((gt - lt) / (len(a) * len(b)))


def boot_ci(a, b, n=None, seed=0):
    n = n or C.N_BOOT
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
         for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    C.ensure_dirs()
    runs = pd.read_csv(os.path.join(C.TRAIN_DIR, "ablation_runs.csv"))
    pi = pd.read_csv(os.path.join(C.XAI_DIR, "xai_per_image.csv"))

    for name, df in (("ablation_runs", runs), ("xai_per_image", pi)):
        hashes = set(df.protocol_hash.dropna().unique())
        assert hashes <= {C.PROTOCOL_HASH}, (
            f"{name} mixes protocols {hashes}; do not pool these rows")

    runs = runs[~runs.get("diverged", pd.Series(False, index=runs.index))
                .fillna(False).astype(bool)]

    # --- T1 dataset --------------------------------------------------------
    mani0 = pd.read_csv(os.path.join(C.DATASET_DIR, "manifest.csv"))
    t1 = (mani0.groupby(["label", "split_i"]).size().unstack(fill_value=0)
          .join(mani0[mani0.split == "test"].groupby("label").size().rename("test")))
    for f, col in (("leakage_removed.csv", "removed_from_train"),
                   ("train_internal_duplicates.csv", "internal_duplicates")):
        p = os.path.join(C.DATASET_DIR, f)
        t1[col] = (pd.read_csv(p).groupby("label").size() if os.path.exists(p)
                   else 0)
    t1 = t1.fillna(0).astype(int)
    t1.to_csv(os.path.join(C.TABLES_DIR, "t1_dataset.csv"))
    print("\n=== T1 dataset ===\n", t1.to_string())

    # --- T2 ablation -------------------------------------------------------
    t2 = (runs.groupby("arch")
          .agg(n=("accuracy", "size"), acc=("accuracy", "mean"), sd=("accuracy", "std"),
               macro_f1=("macro_f1", "mean"), params=("params", "first"),
               best_epoch=("best_epoch", "mean"), hit_cap=("hit_cap", "sum"))
          .round(4).sort_values("acc", ascending=False))
    t2.to_csv(os.path.join(C.TABLES_DIR, "t2_ablation.csv"))
    print("\n=== T2 ablation ===\n", t2.to_string())
    if t2.hit_cap.sum():
        print("WARNING: some runs hit the epoch cap; they were still improving.")

    # --- T3 McNemar --------------------------------------------------------
    preds, rows = {}, []
    for _, r in runs.iterrows():
        f = os.path.join(C.TRAIN_DIR, "histories", f"{r['run_id']}_pred.npy")
        if os.path.exists(f):
            preds.setdefault(r["arch"], []).append(np.load(f))
    mani = pd.read_csv(os.path.join(C.DATASET_DIR, "manifest.csv"))
    true = mani[mani.split == "test"].y.values
    archs = [a for a in C.ARCHS if a in preds]
    for i, a in enumerate(archs):
        for b in archs[i + 1:]:
            pa, pb = preds[a][0] == true, preds[b][0] == true
            x, y = int((pa & ~pb).sum()), int((~pa & pb).sum())
            p = binomtest(min(x, y), x + y, 0.5).pvalue if x + y else 1.0
            rows.append(dict(a=a, b=b, a_only=x, b_only=y, p_raw=p))
    t3 = pd.DataFrame(rows)
    if len(t3):
        t3["p_holm"] = holm(t3.p_raw)
    t3.to_csv(os.path.join(C.TABLES_DIR, "t3_mcnemar.csv"), index=False)
    print("\n=== T3 McNemar ===\n", t3.round(5).to_string(index=False))

    # --- T4 alignment ------------------------------------------------------
    cols = ["auc_intrabrain", "auc_full", "auc_upsampled", "auc_pre_relu_ib",
            "energy_lift_brain",
            "spearman", "dice", "null_dice", "dice_minus_null", "uniform_floor",
            "coverage", "precision", "ceiling",
            "heatmap_area", "lesion_area", "background_fraction",
            "cam_mean_background", "cam_mean_brain_nonlesion", "cam_mean_lesion"]
    t4 = pi.groupby(["arch", "role", "true_class"])[cols].mean().round(4)
    t4["spearman_nan_rate"] = (pi.groupby(["arch", "role", "true_class"])
                               .spearman.apply(lambda s: s.isna().mean()).round(3))
    t4.to_csv(os.path.join(C.TABLES_DIR, "t4_alignment.csv"))
    print("\n=== T4 alignment (primary role only) ===")
    print(t4.xs(C.PRIMARY_ROLE, level="role")[
        ["auc_intrabrain", "auc_full", "energy_lift_brain", "dice"]].to_string())
    print("\n=== T4b Dice against its own baselines (primary role) ===")
    print("  null_dice: within-class permuted mask. uniform_floor: 2p/(1+p), a map")
    print("  active everywhere. ceiling: 2*min(H,M)/(H+M), the most this map size")
    print("  could score. A Dice is not interpretable without them.")
    print(t4.xs(C.PRIMARY_ROLE, level="role")[
        ["dice", "null_dice", "dice_minus_null", "uniform_floor", "ceiling"]].to_string())

    # --- T5 dissociation ---------------------------------------------------
    rows = []
    for (arch, role), g in pi.groupby(["arch", "role"]):
        for i, ca in enumerate(C.TUMOR_CLASSES):
            for cb in C.TUMOR_CLASSES[i + 1:]:
                a = g[g.true_class == ca][C.PRIMARY_METRIC].dropna().values
                b = g[g.true_class == cb][C.PRIMARY_METRIC].dropna().values
                if len(a) < 5 or len(b) < 5:
                    continue
                lo, hi = boot_ci(a, b, seed=C.ANALYSIS_SEED)
                rows.append(dict(arch=arch, role=role, a=ca, b=cb,
                                 mean_a=a.mean(), mean_b=b.mean(),
                                 diff=a.mean() - b.mean(), ci_lo=lo, ci_hi=hi,
                                 cliffs_delta=cliffs_delta(a, b),
                                 p_raw=mannwhitneyu(a, b, alternative="two-sided").pvalue))
    t5 = pd.DataFrame(rows)
    if len(t5):
        t5["p_holm"] = holm(t5.p_raw)
    t5.to_csv(os.path.join(C.TABLES_DIR, "t5_dissociation.csv"), index=False)

    # --- T6 controls -------------------------------------------------------
    src = os.path.join(C.XAI_DIR, "instrument_controls.csv")
    if os.path.exists(src):
        pd.read_csv(src).to_csv(os.path.join(C.TABLES_DIR, "t6_controls.csv"), index=False)

    # --- T7 stability across seeds ----------------------------------------
    per_seed = (pi.groupby(["arch", "role", "seed", "true_class"])[C.PRIMARY_METRIC]
                .mean().unstack())
    rows = []
    for (arch, role), g in per_seed.groupby(level=[0, 1]):
        lowest = [row.idxmin() for _, row in g.iterrows()]
        rows.append(dict(arch=arch, role=role, lowest_per_seed="|".join(lowest),
                         stable=len(set(lowest)) == 1,
                         spread=float((g.max() - g.min()).max())))
    t7 = pd.DataFrame(rows)
    t7.to_csv(os.path.join(C.TABLES_DIR, "t7_stability.csv"), index=False)
    print("\n=== T7 stability of the class ordering across seeds ===")
    print(t7.round(3).to_string(index=False))
    st = t7[t7.stable]
    print(f"\n  stable in {len(st)}/{len(t7)} cells; lowest class there: "
          f"{set(st.lowest_per_seed.str.split('|').str[0]) if len(st) else '-'}")

    # --- T8 what the comparison set costs ---------------------------------
    t8 = (pi.groupby("true_class")
          .agg(bg_frac=("background_fraction", "mean"),
               auc_full=("auc_full", "mean"),
               auc_intrabrain=("auc_intrabrain", "mean")).round(4))
    t8["max_tie_credit"] = (0.5 * t8.bg_frac).round(4)
    t8["full_minus_intrabrain"] = (t8.auc_full - t8.auc_intrabrain).round(4)
    t8.to_csv(os.path.join(C.TABLES_DIR, "t8_comparison_set.csv"))
    print("\n=== T8 the comparison set ===\n", t8.to_string())
    print("\n=== T8b what bilinear upsampling adds, by class ===")
    # auc_full and auc_upsampled share the whole-frame comparison set, so the only
    # difference between them is the interpolation. Comparing against auc_intrabrain
    # would confound resolution with the comparison set.
    up = (pi[pi.role == C.PRIMARY_ROLE].groupby(["arch", "true_class"])
          .agg(native=("auc_full", "mean"), upsampled=("auc_upsampled", "mean"),
               map_area=("heatmap_area", "mean")).round(4))
    up["inflation"] = (up.upsampled - up.native).round(4)
    print(up.to_string())
    r = np.corrcoef(up.map_area, up.inflation)[0, 1]
    print(f"\n  inflation positive in {int((up.inflation > 0).sum())}/{len(up)} cells, "
          f"mean {up.inflation.mean():+.4f}, range {up.inflation.min():+.4f} to "
          f"{up.inflation.max():+.4f}")
    print(f"  correlation with binarised map area: {r:+.2f} - compact maps gain most,")
    print("  so an upsampled between-class comparison partly measures compactness.")

    print("\n  Background fraction differs by class, so full-frame AUC hands one class")
    print("  an advantage before any model behaviour is measured. This is why the")
    print("  intra-brain comparison set is used throughout.")
    print("\nall tables ->", C.TABLES_DIR)


if __name__ == "__main__":
    main()
