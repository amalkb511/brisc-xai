"""
E2 - Summary of the external pituitary check (E1) next to the BRISC pituitary values.

Reads E1_per_image_<tag>.csv (external slices) and data_tables/xai_per_image.csv (BRISC, 20 runs,
true class pituitary, primary layer 'deep'), and writes E2_summary_<tag>.csv:
  per architecture: sensitivity (recall on pituitary), Dice, permuted-mask null, Dice minus null,
  uniform floor, ceiling, intra-brain AUC and energy lift, with 95% hierarchical-bootstrap
  intervals (external: patients -> slices within patient -> seeds within slice;
  BRISC: images -> seeds within image, as in A8).
Also E2_by_plane_<tag>.csv and E2_correct_vs_wrong_<tag>.csv for the external set.

Usage: python E2_external_summary.py --e1 E1_per_image_primary.csv --brisc data_tables/xai_per_image.csv --out out_dir
"""
import argparse
import os

import numpy as np
import pandas as pd

METRICS = ["correct", "dice", "null_dice", "dice_minus_null", "uniform_floor", "ceiling",
           "auc_intrabrain", "energy_lift_brain"]
B = 2000
RNG = np.random.default_rng(20261020)


def wide(df, unit_cols, metric):
    """-> (units, seeds) matrix of the metric. unit_cols identify a unit (slice or image)."""
    p = df.pivot_table(index=unit_cols, columns="seed", values=metric, aggfunc="mean")
    return p


def boot_ci_external(df, metric, B=B):
    """patients -> slices (within patient) -> seeds (within slice)."""
    p = wide(df, ["pid", "name"], metric)
    groups = [g.values for _, g in p.groupby(level="pid")]
    vals = []
    for _ in range(B):
        pid_idx = RNG.integers(0, len(groups), len(groups))
        tot, cnt = 0.0, 0
        for k in pid_idx:
            g = groups[k]
            sl = RNG.integers(0, len(g), len(g))
            sub = g[sl]                                         # (n_slices, n_seeds)
            se = RNG.integers(0, sub.shape[1], sub.shape)       # seeds resampled within slice
            x = np.take_along_axis(sub, se, axis=1)
            x = x[~np.isnan(x)]
            tot += x.sum(); cnt += x.size
        vals.append(tot / cnt)
    return np.percentile(vals, [2.5, 97.5])


def boot_ci_images(df, metric, B=B):
    """images -> seeds within image (A8-style)."""
    p = wide(df, ["stem"], metric).values
    n = len(p)
    vals = []
    for _ in range(B):
        im = RNG.integers(0, n, n)
        sub = p[im]
        se = RNG.integers(0, sub.shape[1], sub.shape)
        x = np.take_along_axis(sub, se, axis=1)
        x = x[~np.isnan(x)]
        vals.append(x.mean())
    return np.percentile(vals, [2.5, 97.5])


def summarize(ext, bri):
    rows = []
    for arch in ["Baseline", "CNN_GAP", "CNN_GAP_BN", "Proposed"]:
        e = ext[ext.arch == arch]
        b = bri[(bri.arch == arch) & (bri.role == "deep") & (bri.true_class == "pituitary")].copy()
        b["correct"] = np.nan   # accuracy per image is not stored in xai_per_image.csv
        for met in METRICS:
            r = dict(arch=arch, metric=met)
            if len(e):
                lo, hi = boot_ci_external(e, met) if met != "correct" else boot_ci_external(e.assign(correct=e.correct.astype(float)), met)
                r.update(ext_mean=e[met].astype(float).mean(), ext_lo=lo, ext_hi=hi,
                         ext_n_slices=e.name.nunique(), ext_n_patients=e.pid.nunique())
            if met in b.columns and b[met].notna().any():
                lo, hi = boot_ci_images(b, met)
                r.update(brisc_mean=b[met].mean(), brisc_lo=lo, brisc_hi=hi, brisc_n_images=b.stem.nunique())
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--brisc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="primary")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ext = pd.read_csv(a.e1)
    bri = pd.read_csv(a.brisc)
    if "dice_minus_null" not in bri.columns:
        bri["dice_minus_null"] = bri.dice - bri.null_dice
    s = summarize(ext, bri)
    s.to_csv(os.path.join(a.out, f"E2_summary_{a.tag}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))

    # Secondary breakdowns on the external set (descriptive; seeds pooled)
    g = ext.groupby(["arch", "plane"])[["correct", "dice", "null_dice", "dice_minus_null", "auc_intrabrain"]].mean()
    g["n_slices"] = ext.groupby(["arch", "plane"]).name.nunique()
    g.round(4).to_csv(os.path.join(a.out, f"E2_by_plane_{a.tag}.csv"))
    cw = ext.groupby(["arch", "correct"])[["dice", "null_dice", "dice_minus_null", "auc_intrabrain"]].mean()
    cw["n_rows"] = ext.groupby(["arch", "correct"]).size()
    cw.round(4).to_csv(os.path.join(a.out, f"E2_correct_vs_wrong_{a.tag}.csv"))
    print("\nBy plane:\n", g.round(4).to_string())
    print("\nCorrect vs wrong:\n", cw.round(4).to_string())


if __name__ == "__main__":
    main()
