#!/usr/bin/env python3
"""B4d - Robustness check of the peak definition (pituitary, deep layer, GPU, a few minutes).
B4b defined the peak = centroid of the region >= 90 % of the max. Here:
  peak_y, peak_x = pixel of the MAXIMUM (in case of ties: centroid of the pixels >= 0.999 * max), plus n_top_components =
  number of connected zones of the region >= 90 % of the max, and dist_argmax_vs_centroid = gap between the two definitions.
Output: B4d_per_image.csv, in the SAME format as B4b_per_image.csv for the columns used by B4c
  -> then: python B4c_peak_tests_v3.py --b4b <out>/B4d_per_image.csv --manifest ... --out <out>/B4c_summary_argmax.csv
Usage: cd /content/brisc-xai && BRISC_OUT=... python -u B4d_argmax_control.py --out $BRISC_OUT/B4
"""
import argparse, os, time
import cv2, numpy as np, pandas as pd

S = 224
def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([ys.mean(), xs.mean()]) if len(ys) else np.array([np.nan, np.nan])

def describe(up, post_max, msk):
    if post_max <= 1e-12:
        return dict(degenerate=True, peak_y=np.nan, peak_x=np.nan, cen_y=np.nan, cen_x=np.nan, n_top_components=np.nan,
                    top_area=np.nan, dist_argmax_vs_centroid=np.nan, peak_in_lesion=np.nan)
    mx = up.max()
    pk = centroid(up >= 0.999 * mx)                       # argmax (centroid of the ties)
    top = up >= 0.9 * mx
    ce = centroid(top)                                    # B4b definition
    n_cc = cv2.connectedComponents(top.astype(np.uint8), connectivity=8)[0] - 1
    iy, ix = min(max(int(round(pk[0])), 0), S - 1), min(max(int(round(pk[1])), 0), S - 1)
    return dict(degenerate=False, peak_y=pk[0], peak_x=pk[1], cen_y=ce[0], cen_x=ce[1], n_top_components=int(n_cc),
                top_area=float(top.mean()), dist_argmax_vs_centroid=float(np.linalg.norm(pk - ce) / S),
                peak_in_lesion=bool(msk[iy, ix]))

def main():
    import config as C
    from _xai_common import load_inputs, iter_models, roles_for
    import _xai_common as X
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); ap.add_argument("--cls", default="pituitary")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    xai, IMGS, MSK, LBL, CIX, BRAIN = load_inputs()
    idx = np.where(LBL == a.cls)[0]; print(f"{a.cls}: {len(idx)} images", flush=True)
    I, M, Y = IMGS[idx], MSK[idx], CIX[idx]; stems = xai.stem.values[idx]
    rows = []
    for arch, seed, rid, model in iter_models():
        for role, layer in roles_for(arch):
            raw, post, up = X.cam_continuous(model, layer, I, Y)
            pmax = post.max(axis=(1, 2))
            for i in range(len(idx)):
                d = describe(up[i], pmax[i], M[i]); d.update(arch=arch, seed=seed, role=role, layer=layer, stem=stems[i]); rows.append(d)
        pd.DataFrame(rows).to_csv(f"{a.out}/B4d_per_image.csv", index=False)
        print(f"  {rid} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    d = pd.DataFrame(rows); d.to_csv(f"{a.out}/B4d_per_image.csv", index=False); nd = d[~d.degenerate]
    s = nd.groupby(["arch", "role"]).agg(n_obs=("stem", "size"), frac_multi_component=("n_top_components", lambda v: (v > 1).mean()),
        n_components_med=("n_top_components", "median"), n_components_max=("n_top_components", "max"),
        dist_argmax_vs_centroid_med=("dist_argmax_vs_centroid", "median"),
        dist_argmax_vs_centroid_q3=("dist_argmax_vs_centroid", lambda v: v.quantile(.75)),
        peak_in_lesion_frac=("peak_in_lesion", "mean"), top_area_mean=("top_area", "mean")).reset_index().round(4)
    s.to_csv(f"{a.out}/B4d_summary.csv", index=False); print(s.set_index(["arch", "role"]).T.to_string()); print("wrote B4d_* in", a.out)

if __name__ == "__main__": main()
