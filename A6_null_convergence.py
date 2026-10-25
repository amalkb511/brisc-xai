#!/usr/bin/env python3
"""
A6 — Convergence of the permuted null (Eq. 31) at R = 5, 100, 500, 1000 (FAST VERSION, vectorised).

Same definition as the original version: for each image i, the null is the Dice of the SAME binarised
map (up[i] >= C.THRESHOLD, path identical to 03_measure_xai.py) against the mask of ANOTHER
image j != i of the same class, averaged over R uniform draws; then averaged over images.

Implementation difference only: instead of R x n calls to dice(), the matrix D[i,j] = Dice(map_i, mask_j)
is computed ONCE by matrix product (checked identical to dice_bin, error 0), and the j are then
drawn with the same law (uniform on {0..n-1}\\{i}). Checked on synthetic data: same mean
and same standard deviation as the original loop.

OUTPUT: CSV (arch, role, class, R, null_mean, ci95_lo, ci95_hi, delta_vs_R5), identical to the previous version.
  null_mean = mean over model seeds x RNG_SEEDS; 95% CI = percentiles 2.5/97.5 over these values.
The CSV is rewritten after EACH model (partial result usable if the session is interrupted).
Run :  python -u A6_null_convergence.py --out A6_null_convergence.csv
"""
import argparse, collections, time
import numpy as np, pandas as pd
import config as C
from _xai_common import load_inputs, iter_models, roles_for, cam_continuous

R_LIST = [5, 100, 500, 1000]
RNG_SEEDS = [0, 1, 2]


def dice_matrix(B, M):
    """B, M : n x P (0/1 float32). D[i,j] = Dice(B_i, M_j) ; 0 if both are empty (as in dice_bin)."""
    inter = B @ M.T
    denom = B.sum(1)[:, None] + M.sum(1)[None, :]
    return np.where(denom > 0, 2.0 * inter / np.maximum(denom, 1e-12), 0.0)


def null_fast(D, R, rng):
    n = D.shape[0]
    j = rng.integers(0, n - 1, size=(n, R))
    j = j + (j >= np.arange(n)[:, None])           # skip i: uniform over the other images
    return float(D[np.arange(n)[:, None], j].mean(axis=1).mean())


def build_rows(acc):
    rows, base = [], {}
    for (arch, role, cls, R), vals in sorted(acc.items()):
        v = np.array(vals); m = float(v.mean())
        lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) > 1 else (m, m)
        if R == R_LIST[0]:
            base[(arch, role, cls)] = m
        rows.append(dict(arch=arch, role=role, **{"class": cls}, R=R,
                         null_mean=round(m, 4), ci95_lo=round(float(lo), 4),
                         ci95_hi=round(float(hi), 4),
                         delta_vs_R5=round(m - base.get((arch, role, cls), m), 4)))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="A6_null_convergence.csv")
    a = ap.parse_args()
    t0 = time.time()
    _, IMGS, MSK, LBL, CIX, _ = load_inputs()
    print("inputs charges en %.1f min" % ((time.time() - t0) / 60), flush=True)

    classes = list(np.unique(LBL))
    sel = {c: [i for i in range(len(IMGS)) if LBL[i] == c] for c in classes}
    # mask matrices per class (n_c x P), computed once
    Mmat = {c: MSK[sel[c]].reshape(len(sel[c]), -1).astype(np.float32) for c in classes}

    acc = collections.defaultdict(list)   # (arch, role, class, R) -> values (model seed x rng)

    for arch, seed, rid, model in iter_models():
        for role, layer in roles_for(arch):
            _, post, up = cam_continuous(model, layer, IMGS, CIX)
            binm = (up >= C.THRESHOLD)
            for cls in classes:
                B = binm[sel[cls]].reshape(len(sel[cls]), -1).astype(np.float32)
                D = dice_matrix(B, Mmat[cls])
                for R in R_LIST:
                    for s in RNG_SEEDS:
                        acc[(arch, role, cls, R)].append(
                            null_fast(D, R, np.random.default_rng(s + 1000 * seed)))
        pd.DataFrame(build_rows(acc)).to_csv(a.out, index=False)   # partial save
        print("  %s done  (%.1f min elapsed)" % (rid, (time.time() - t0) / 60), flush=True)

    pd.DataFrame(build_rows(acc)).to_csv(a.out, index=False)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
