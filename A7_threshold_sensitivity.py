#!/usr/bin/env python3
"""
A7 — Sensitivity of the Dice to the threshold, RECOMPUTING the maps through the same path as 03_measure_xai.py
(models.CAM.raw -> ReLU -> upsample+max-norm). No .npy files are read.

The Dice reproduces that of 03_measure_xai (full frame, max-normalised 'up' map thresholded against the mask MSK).
For the robustness requested by reviewer A7 we add:
  * Dice at several thresholds tau (TAUS),
  * Dice maximised over tau (no fixed threshold),
  * area-matched Dice (binarised to the same number of pixels as the mask).
The threshold-free AUC is already in Table 10 (auc_intrabrain): no need to recompute it here.

Degenerate maps (post.max<=1e-12): counted as zero overlap (as in production) and tallied.

OUTPUT: A7_threshold.csv  -> columns (arch, role, class, metric, value, n_images, n_degenerate)
         values averaged over images AND seeds, per (arch, role, class), as in Fig. 4 / Table 10.
Run :  python -I A7_threshold_sensitivity.py --out A7_threshold.csv
"""
import argparse, collections
import numpy as np, pandas as pd
import config as C
from _xai_common import load_inputs, iter_models, roles_for, cam_continuous, dice_bin

TAUS = [0.3, 0.4, 0.5, 0.6, 0.7]
TAU_GRID = np.linspace(0.05, 0.95, 19)


def dice_area_matched(cont, mask):
    k = int(mask.sum())
    if k == 0:
        return 0.0
    thr = np.sort(cont.ravel())[::-1][k - 1]      # the k strongest pixels
    return dice_bin(cont >= thr, mask)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="A7_threshold.csv")
    a = ap.parse_args()
    _, IMGS, MSK, LBL, CIX, _ = load_inputs()

    # accumulators per (arch, role, class, metric) -> list of per-image values (all seeds)
    acc = collections.defaultdict(list)
    ndeg = collections.Counter(); nimg = collections.Counter()

    for arch, seed, rid, model in iter_models():
        for role, layer in roles_for(arch):
            _, post, up = cam_continuous(model, layer, IMGS, CIX)
            for i in range(len(IMGS)):
                cls = LBL[i]; key = (arch, role, cls)
                nimg[key] += 1
                degenerate = bool(post[i].max() <= 1e-12)   # production convention
                if degenerate:
                    ndeg[key] += 1
                m = MSK[i]
                for tau in TAUS:
                    acc[(arch, role, cls, f"dice_tau={tau}")].append(dice_bin(up[i] >= tau, m))
                acc[(arch, role, cls, "dice_max")].append(max(dice_bin(up[i] >= t, m) for t in TAU_GRID))
                acc[(arch, role, cls, "dice_area_matched")].append(dice_area_matched(up[i], m))
        print(" ", rid, "done")

    rows = []
    for (arch, role, cls, metric), vals in acc.items():
        key = (arch, role, cls)
        rows.append(dict(arch=arch, role=role, **{"class": cls}, metric=metric,
                         value=round(float(np.mean(vals)), 4),
                         n_images=nimg[key], n_degenerate=ndeg[key]))
    pd.DataFrame(rows).sort_values(["arch", "role", "class", "metric"]).to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
