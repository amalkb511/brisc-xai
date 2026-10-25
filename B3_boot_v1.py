#!/usr/bin/env python3
"""B3 - per-class metrics + 95% CI by IMAGE BOOTSTRAP (same image resample for the 5 seeds), + confusion matrices.
Input: --pred CSV with one row per (architecture, seed, image): columns image / architecture / seed / y_true / y_pred (names detected automatically).
Output (in --out-dir): B3_perclass_boot.csv (arch, class, metric, mean_over_seeds, sd_seeds, ci95_lo, ci95_hi),
                          B3_confusion_<arch>.csv (mean counts over seeds).
CI = percentiles 2.5/97.5 of the seed-averaged metrics over 2000 image resamples. Seed 0 fixed.
Usage: python -u B3_boot_v1.py --pred <pred.csv> --out-dir <folder>
"""
import argparse, os, numpy as np, pandas as pd
P = dict(image=["image_id", "stem", "image", "path", "filename", "name"], arch=["architecture", "arch", "model"], seed=["seed"],
         yt=["y_true", "true", "label", "true_class"], yp=["y_pred", "pred", "prediction", "pred_class"])
def col(df, k):
    for c in P[k]:
        if c in df.columns: return c
    raise SystemExit(f"column '{k}' not found among {list(df.columns)}")
def metrics(cm):
    tot = cm.sum(); out = {}
    tp = np.diag(cm).astype(float); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp; tn = tot - tp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = tp / (tp + fn); spec = tn / (tn + fp); prec = tp / (tp + fp); f1 = 2 * prec * sens / (prec + sens)
    return dict(sensitivity=sens, specificity=spec, precision=prec, f1=f1)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--out-dir", default="."); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True); df = pd.read_csv(a.pred)
    ci, ca, cs, ct, cp = [col(df, k) for k in ("image", "arch", "seed", "yt", "yp")]; print("columns:", ci, ca, cs, ct, cp)
    labs = sorted(pd.unique(df[[ct, cp]].values.ravel())); L = len(labs); m = {l: i for i, l in enumerate(labs)}; print("classes:", labs)
    rng = np.random.default_rng(0); rows = []
    for arch, ga in df.groupby(ca):
        seeds = sorted(ga[cs].unique()); imgs = sorted(ga[ci].unique()); n = len(imgs)
        T = np.zeros((len(seeds), n), int); Pm = np.zeros((len(seeds), n), int)
        for s, sd in enumerate(seeds):
            g = ga[ga[cs] == sd].drop_duplicates(ci).set_index(ci).loc[imgs]; T[s] = g[ct].map(m).values; Pm[s] = g[cp].map(m).values
        def cm_of(idx, s): return np.bincount(T[s, idx] * L + Pm[s, idx], minlength=L * L).reshape(L, L)
        full = np.arange(n); pt = {k: np.mean([metrics(cm_of(full, s))[k] for s in range(len(seeds))], 0) for k in ("sensitivity", "specificity", "precision", "f1")}
        sdv = {k: np.std([metrics(cm_of(full, s))[k] for s in range(len(seeds))], 0, ddof=1) for k in pt}
        bs = {k: [] for k in pt}
        for _ in range(a.B):
            idx = rng.integers(0, n, n); r = [metrics(cm_of(idx, s)) for s in range(len(seeds))]
            for k in bs: bs[k].append(np.mean([x[k] for x in r], 0))
        for k in pt:
            b = np.array(bs[k]); lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)
            for j, l in enumerate(labs): rows.append(dict(architecture=arch, **{"class": l}, metric=k, mean_over_seeds=round(pt[k][j], 4), sd_seeds=round(sdv[k][j], 4), ci95_lo=round(lo[j], 4), ci95_hi=round(hi[j], 4)))
        pd.DataFrame(np.mean([cm_of(full, s) for s in range(len(seeds))], 0), index=labs, columns=labs).round(2).to_csv(os.path.join(a.out_dir, f"B3_confusion_{arch}.csv"))
        print(arch, "seeds", seeds, "n images", n)
    o = pd.DataFrame(rows); o.to_csv(os.path.join(a.out_dir, "B3_perclass_boot.csv"), index=False)
    print(o[o.metric.isin(["sensitivity", "precision"])].pivot_table(index=["architecture", "class"], columns="metric", values=["mean_over_seeds", "ci95_lo", "ci95_hi"]).round(3).to_string())
if __name__ == "__main__": main()
