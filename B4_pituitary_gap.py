#!/usr/bin/env python3
"""B4 - accuracy / localisation gap (pituitary). CPU, ~1 min, no model loaded.
Inputs: xai/xai_per_image.csv + B3_input_pred.csv (image_id, architecture, seed, y_true, y_pred)
Outputs: B4_correct_vs_wrong.csv, B4_size.csv, B4_resolution.csv, B4_context.csv
"""
import argparse, os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, rankdata

ALIAS = {"CNN_GAP_BN_SE": "Proposed", "Baseline_CNN": "Baseline"}
METR = ["auc_intrabrain", "dice", "energy_lift_brain"]
rng = np.random.default_rng(20261018)

def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = rankdata(np.r_[a, b]); u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return 2 * u / (len(a) * len(b)) - 1

def boot_diff(df, col, flag, B=2000):
    """difference of means (correct - wrong), bootstrap by IMAGE (all seeds of an image together)."""
    df = df.assign(_f=np.asarray(flag, bool), _w=np.asarray(df[col], float))
    df["_c1"] = df._w * df._f; df["_c0"] = df._w * (~df._f)
    df["_n1"] = df._f.astype(float); df["_n0"] = 1.0 - df._n1
    s = df.groupby("stem")[["_c1", "_n1", "_c0", "_n0"]].sum().to_numpy()
    out = []
    for _ in range(B):
        t = s[rng.integers(0, len(s), len(s))].sum(0)
        if t[1] > 0 and t[3] > 0: out.append(t[0] / t[1] - t[2] / t[3])
    return np.nanpercentile(out, [2.5, 97.5])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xai", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="."); ap.add_argument("--cls", default="pituitary")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    x = pd.read_csv(a.xai); p = pd.read_csv(a.pred)
    p["arch"] = p["architecture"].replace(ALIAS)
    idcol = "image_id" if "image_id" in p else "stem"
    p["stem"] = p[idcol].astype(str).map(lambda s: os.path.splitext(os.path.basename(s))[0])
    p["correct"] = p["y_true"] == p["y_pred"]
    d = x[x.true_class == a.cls].merge(p[["arch", "seed", "stem", "correct", "y_pred"]],
                                       on=["arch", "seed", "stem"], how="left")
    print(f"rows {a.cls}: {len(d)} | without prediction: {d.correct.isna().sum()}")
    assert d.correct.notna().all(), "jointure arch/seed/stem incomplete"
    d["correct"] = d.correct.astype(bool)
    d["lesion_pct"] = 100 * d.lesion_area
    d["dice_over_ceiling"] = d.dice / d.ceiling.clip(lower=1e-9)
    d["ratio_les_nonles"] = d.cam_mean_lesion / d.cam_mean_brain_nonlesion.clip(lower=1e-12)

    # 1. correct vs wrong
    rows = []
    for (arch, role), g in d.groupby(["arch", "role"]):
        for m in METR:
            c, w = g[g.correct][m].dropna(), g[~g.correct][m].dropna()
            if len(w) < 5: continue
            lo, hi = boot_diff(g.assign(_v=g[m].fillna(0)), "_v", g.correct.values)
            rows.append(dict(arch=arch, role=role, metric=m, n_correct=len(c), n_wrong=len(w),
                             n_wrong_images=g[~g.correct].stem.nunique(),
                             mean_correct=c.mean(), mean_wrong=w.mean(), diff=c.mean() - w.mean(),
                             ci95_lo=lo, ci95_hi=hi, mwu_p=mannwhitneyu(c, w).pvalue, cliffs=cliffs(c, w)))
    pd.DataFrame(rows).round(4).to_csv(f"{a.out}/B4_correct_vs_wrong.csv", index=False)

    # 2. lesion size
    rows = []
    for (arch, role), g in d.groupby(["arch", "role"]):
        gm = g.groupby("stem")[METR + ["lesion_pct"]].mean()      # mean over seeds: 1 point / image
        q = pd.qcut(gm.lesion_pct, 3, labels=["petit", "moyen", "grand"])
        for m in METR:
            r, pv = spearmanr(gm.lesion_pct, gm[m], nan_policy="omit")
            rows.append(dict(arch=arch, role=role, metric=m, n_images=len(gm), spearman=r, p=pv,
                             **{f"mean_{k}": gm[m][q == k].mean() for k in ["petit", "moyen", "grand"]},
                             lesion_pct_median=gm.lesion_pct.median(),
                             lesion_pct_q1=gm.lesion_pct.quantile(.25), lesion_pct_q3=gm.lesion_pct.quantile(.75)))
    pd.DataFrame(rows).round(4).to_csv(f"{a.out}/B4_size.csv", index=False)

    # 3. resolution: Dice ceiling imposed by the grid vs observed Dice, deep (28) vs head (14)
    rows = []
    for (arch, role), g in d.groupby(["arch", "role"]):
        rows.append(dict(arch=arch, role=role, resolution=g.resolution.iloc[0],
                         ceiling_median=g.ceiling.median(), ceiling_q1=g.ceiling.quantile(.25), ceiling_q3=g.ceiling.quantile(.75),
                         dice_mean=g.dice.mean(), dice_median=g.dice.median(),
                         null_dice_mean=g.null_dice.mean(), dice_minus_null_mean=g.dice_minus_null.mean(),
                         frac_dice_gt_null=(g.dice_minus_null > 0).mean(),
                         dice_over_ceiling_mean=g.dice_over_ceiling.mean(),
                         frac_ceiling_ge_0_5=(g.ceiling >= .5).mean(), degenerate_frac=g.degenerate.mean(),
                         heatmap_area_mean=g.heatmap_area.mean(), lesion_area_mean=g.lesion_area.mean()))
    pd.DataFrame(rows).round(4).to_csv(f"{a.out}/B4_resolution.csv", index=False)

    # 4. context: where does the saliency go? (mean CAM per zone; >1 = favours the lesion)
    rows = []
    for (arch, role), g in d.groupby(["arch", "role"]):
        # labels kept as in the stored B4_context.csv: "tous" = all, "corrects" = correct, "faux" = wrong
        for lab, h in [("tous", g), ("corrects", g[g.correct]), ("faux", g[~g.correct])]:
            if len(h) < 5: continue
            rows.append(dict(arch=arch, role=role, groupe=lab, n=len(h),
                             cam_lesion=h.cam_mean_lesion.mean(), cam_brain_nonlesion=h.cam_mean_brain_nonlesion.mean(),
                             cam_background=h.cam_mean_background.mean(),
                             ratio_lesion_nonlesion_median=h.ratio_les_nonles.median(),
                             frac_lesion_below_nonlesion=(h.cam_mean_lesion < h.cam_mean_brain_nonlesion).mean(),
                             background_fraction_mean=h.background_fraction.mean(),
                             cam_bg_over_brain=(h.cam_mean_background / h.cam_mean_brain_nonlesion.clip(lower=1e-12)).median()))
    pd.DataFrame(rows).round(4).to_csv(f"{a.out}/B4_context.csv", index=False)
    print("wrote B4_*.csv in", a.out)

if __name__ == "__main__": main()
