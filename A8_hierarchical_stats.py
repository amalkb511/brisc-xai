#!/usr/bin/env python3
"""
A8 (v2, vectorised, reads xai_per_image.csv directly) — between-class contrasts with hierarchical bootstrap.

Replaces Mann-Whitney U (which treats each image x seed row as independent) by a TWO-level bootstrap:
  level 1: resample the IMAGES (clusters) within each class;
  level 2: for each drawn image, resample its measurements over the SEEDS (with replacement).
The contrast (difference of means between two classes) is computed per architecture and per metric; the architectures
are handled separately (stratified analysis). Holm is applied per (architecture, metric) over the 3 class contrasts.

INPUT: --xai  xai_per_image.csv  (columns used: arch, seed, role, true_class, stem, + metrics)
OUTPUT: --out CSV (architecture, metric, contrast, n_img_A, n_img_B, diff, ci95_lo, ci95_hi, p_raw, p_holm, cliffs_delta)
  p_raw = 2*min(P(boot<=0), P(boot>=0)) (two-sided, percentile bootstrap). p = 0.0000 means p < 1/B (no draw of opposite sign
  out of B resamples): report 'p < 0.001', not 'p = 0'.
  cliffs_delta = effect size on the per-image mean values (not on the rows), descriptive only.
Run : python -u A8_hierarchical_stats.py --xai .../xai/xai_per_image.csv --out A8_hier.csv
"""
import argparse, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata

METRICS = {"AUC": "auc_intrabrain", "Dice": "dice", "EnergyLift": "energy_lift_brain"}
CLASSES = ["glioma", "meningioma", "pituitary"]
B = 5000


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def boot_mean(V, rng, B):
    """V : n_images x n_seeds. Returns B two-level bootstrap means."""
    n, s = V.shape
    out = np.empty(B)
    for b in range(B):
        im = rng.integers(0, n, size=n)                  # level 1: images
        sd = rng.integers(0, s, size=(n, s))             # level 2: seeds, per image
        out[b] = V[im[:, None], sd].mean()
    return out


def cliffs(x, y):
    r = rankdata(np.concatenate([x, y])); n1, n2 = len(x), len(y)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return 2 * u / (n1 * n2) - 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--xai", required=True); ap.add_argument("--out", default="A8_hier.csv")
    ap.add_argument("--role", default="deep"); ap.add_argument("--B", type=int, default=B)
    a = ap.parse_args()
    d = pd.read_csv(a.xai); d = d[d.role == a.role]
    rows = []
    for arch, ga in d.groupby("arch"):
        for mname, col in METRICS.items():
            mats, nan_dropped = {}, 0
            for c in CLASSES:
                piv = ga[ga.true_class == c].pivot_table(index="stem", columns="seed", values=col, aggfunc="mean")
                nan_dropped += int(piv.isna().any(axis=1).sum()); piv = piv.dropna()   # images measured on all seeds
                mats[c] = piv.values
            recs = []
            for c1, c2 in itertools.combinations(CLASSES, 2):
                rng = np.random.default_rng(20260918)
                A, Bm = mats[c1], mats[c2]
                obs = A.mean() - Bm.mean()
                boots = boot_mean(A, rng, a.B) - boot_mean(Bm, rng, a.B)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
                recs.append(dict(architecture=arch, metric=mname, contrast=f"{c1}-{c2}",
                                 n_img_A=len(A), n_img_B=len(Bm), diff=round(float(obs), 4),
                                 ci95_lo=round(float(lo), 4), ci95_hi=round(float(hi), 4), p_raw=float(p),
                                 cliffs_delta=round(float(cliffs(A.mean(1), Bm.mean(1))), 3),
                                 n_images_dropped_nan=nan_dropped))
            for r, pa in zip(recs, holm([r["p_raw"] for r in recs])):
                r["p_holm"] = round(float(pa), 4); r["p_raw"] = round(r["p_raw"], 4); rows.append(r)
            print("  ", arch, mname, "done", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False); print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
