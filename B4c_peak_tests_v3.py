#!/usr/bin/env python3
"""B4c (v3) - Quantitative tests on the peak position (CPU, ~1 min, no model).
Inputs: B4/B4b_per_image.csv (peak_y, peak_x, degenerate, arch, role, seed, stem) + dataset/manifest.csv (masks, images)
Output: B4c_summary.csv (per arch x role). Two nulls:
 NULL A "permuted lesion": lesion centroid of ANOTHER image (shared across seeds). Keeps the typical anatomical position.
        -> tests whether the peak follows the lesion of ITS OWN image (one-sided p: closer than the null).
 NULL B "random brain pixel": a pixel drawn uniformly in the brain mask of the same image (one draw per observation).
        -> model-free reference for: distance to the lesion, |dx|, |dy|, share "more lateral than vertical", share at the edge (<= 8 px).
 NB: null B draws an INDEPENDENT pixel per observation (variability underestimated: the seeds of an image are correlated).
        -> *_boot_* columns: 95% CI of the observed statistic by bootstrap BY IMAGE (300 images, with their seeds),
           and p_*_Bboot = share of bootstrap replicates on the "chance" side of B_mean (to be preferred to p_*_B).
 dx = peak_x - lesion_x, dy = peak_y - lesion_y (fraction of the side S). "lateral" = |dx| > |dy|.
"""
import argparse, os, numpy as np, pandas as pd, cv2
from PIL import Image
S = 224; EDGE = 8

def brain_mask(img):                         # identical to _xai_common.brain_mask (img in [0,1], S x S x 3)
    g = (img.mean(axis=2) * 255).astype(np.uint8)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1: return bw > 0
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))

def lesion_centroid(path):
    m = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (S, S), interpolation=cv2.INTER_NEAREST) / 255.0 >= 0.5
    ys, xs = np.nonzero(m); return np.array([ys.mean(), xs.mean()])

def brain_pixels(path):
    img = np.asarray(Image.open(path).convert("RGB").resize((S, S), Image.NEAREST), dtype="float32") / 255.0
    return np.argwhere(brain_mask(img))      # (n,2): y, x

def stats(py, px, cy, cx):
    """py, px, cy, cx: (..., n_obs). Medians/means over the LAST axis."""
    dy, dx = (py - cy) / S, (px - cx) / S
    edge = np.minimum.reduce([py, px, S - 1 - py, S - 1 - px]) <= EDGE
    return dict(dist_med=np.median(np.hypot(dy, dx), -1), abs_dx_med=np.median(np.abs(dx), -1),
                abs_dy_med=np.median(np.abs(dy), -1), frac_lateral=(np.abs(dx) > np.abs(dy)).mean(-1),
                frac_left=(dx < 0).mean(-1), frac_edge=edge.mean(-1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b4b", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="B4c_summary.csv"); ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--cls", default="pituitary"); a = ap.parse_args()
    rng = np.random.default_rng(20261018)
    d = pd.read_csv(a.b4b); d = d[~d.degenerate].copy()
    m = pd.read_csv(a.manifest); m = m[(m.split == "test") & (m.label == a.cls)].drop_duplicates("stem").set_index("stem")
    d = d[d.stem.isin(m.index)].copy()
    stems = np.array(sorted(m.index)); pos = pd.Series(np.arange(len(stems)), index=stems)
    cen = np.stack([lesion_centroid(m.mask_path[s]) for s in stems])
    print("computing brain masks...", flush=True)
    bpx = [brain_pixels(m.path[s]) for s in stems]
    rows = []
    for (arch, role), g in d.groupby(["arch", "role"]):
        py, px = g.peak_y.values, g.peak_x.values; idx = pos[g.stem].values
        obs = {k: float(v) for k, v in stats(py, px, cen[idx, 0], cen[idx, 1]).items()}
        # BOOTSTRAP BY IMAGE of the observed statistic (cluster = image, all its seeds)
        groups = [np.where(idx == st)[0] for st in np.unique(idx)]
        boot = {k: np.empty(a.perm) for k in obs}
        for r in range(a.perm):
            pick = rng.integers(0, len(groups), len(groups)); sel = np.concatenate([groups[j] for j in pick])
            s_ = stats(py[sel], px[sel], cen[idx[sel], 0], cen[idx[sel], 1])
            for k in boot: boot[k][r] = s_[k]
        # NULL A: centroids shuffled across images
        A = {k: np.empty(a.perm) for k in obs}
        for r in range(a.perm):
            perm = rng.permutation(len(stems)); s = stats(py, px, cen[perm[idx], 0], cen[perm[idx], 1])
            for k in A: A[k][r] = s[k]
        # NULL B: one random brain pixel per observation, a.perm repetitions (vectorised)
        by = np.empty((a.perm, len(g))); bx = np.empty((a.perm, len(g)))
        for st in np.unique(idx):
            cols = np.where(idx == st)[0]; pts = bpx[st]
            pick = rng.integers(0, len(pts), size=(a.perm, len(cols)))
            by[:, cols] = pts[pick, 0]; bx[:, cols] = pts[pick, 1]
        B = stats(by, bx, cen[idx, 0][None, :], cen[idx, 1][None, :])
        row = dict(arch=arch, role=role, n_obs=len(g), n_images=g.stem.nunique())
        for k in obs:
            row.update({f"{k}_obs": obs[k],
                        f"{k}_A_mean": A[k].mean(), f"{k}_A_lo": np.percentile(A[k], 2.5), f"{k}_A_hi": np.percentile(A[k], 97.5),
                        f"{k}_B_mean": B[k].mean(), f"{k}_B_lo": np.percentile(B[k], 2.5), f"{k}_B_hi": np.percentile(B[k], 97.5)})
        for k in obs:
            row.update({f"{k}_boot_lo": np.percentile(boot[k], 2.5), f"{k}_boot_hi": np.percentile(boot[k], 97.5)})
        row["p_closer_than_Bboot"] = (1 + (boot["dist_med"] >= B["dist_med"].mean()).sum()) / (1 + a.perm)
        row["p_more_lateral_than_Bboot"] = (1 + (boot["frac_lateral"] <= B["frac_lateral"].mean()).sum()) / (1 + a.perm)
        row["p_less_lateral_than_Bboot"] = (1 + (boot["frac_lateral"] >= B["frac_lateral"].mean()).sum()) / (1 + a.perm)
        row["p_more_edge_than_Bboot"] = (1 + (boot["frac_edge"] <= B["frac_edge"].mean()).sum()) / (1 + a.perm)
        row["p_closer_than_A"] = (1 + (A["dist_med"] <= obs["dist_med"]).sum()) / (1 + a.perm)
        row["p_closer_than_B"] = (1 + (B["dist_med"] <= obs["dist_med"]).sum()) / (1 + a.perm)
        row["p_more_lateral_than_B"] = (1 + (B["frac_lateral"] >= obs["frac_lateral"]).sum()) / (1 + a.perm)
        row["p_more_edge_than_B"] = (1 + (B["frac_edge"] >= obs["frac_edge"]).sum()) / (1 + a.perm)
        row["edge_chance_uniform_frame"] = 1 - ((S - 2 * EDGE) / S) ** 2
        rows.append(row)
    out = pd.DataFrame(rows).round(4); out.to_csv(a.out, index=False)
    print(out.set_index(["arch", "role"]).T.to_string()); print("wrote", a.out)

if __name__ == "__main__": main()
