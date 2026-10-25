#!/usr/bin/env python3
"""B4b - Where does the saliency fall on pituitary images? (peak, centroids, depth within the brain, mean maps)
Recomputes the maps as in production (via _xai_common.cam_continuous), ONLY on the images of class --cls.
Outputs: B4b_per_image.csv, B4b_summary.csv, B4b_mean_maps.npz, B4b_mean_maps.png
Usage: cd /content/brisc-xai && python -u B4b_saliency_location.py --out $BRISC_OUT/B4 [--roles deep|all]
"""
import argparse, os, time
import cv2, numpy as np, pandas as pd

TOP = 0.9     # "peak" region = pixels of the max-normalised continuous map >= TOP

def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([ys.mean(), xs.mean()]) if len(ys) else np.array([np.nan, np.nan])

def depth_map(brain):
    """distance to the brain edge, normalised by its max: 0 = edge, 1 = deepest point."""
    dt = cv2.distanceTransform(brain.astype(np.uint8), cv2.DIST_L2, 5)
    return dt / max(dt.max(), 1e-9)

def describe(up, post_max, msk, brain, mean_les_c):
    """Localisation metrics of ONE map `up` (S x S, max-normalised). Distances as a fraction of the side S."""
    S = up.shape[0]
    out = dict(degenerate=bool(post_max <= 1e-12))
    keys = ["peak_y", "peak_x", "dist_peak_lesion", "dist_peak_meanlesion", "dist_peak_braincenter",
            "peak_depth", "top_frac_in_lesion", "top_frac_in_brain", "top_area", "peak_in_lesion"]
    if out["degenerate"]:
        out.update({k: np.nan for k in keys}); return out
    top = up >= TOP * up.max()
    pk = centroid(top)                                   # centroid of the peak region (robust to plateaus)
    ly = centroid(msk); bc = centroid(brain)
    dep = depth_map(brain)
    iy, ix = int(round(pk[0])), int(round(pk[1]))
    iy, ix = min(max(iy, 0), S - 1), min(max(ix, 0), S - 1)
    out.update(peak_y=pk[0], peak_x=pk[1],
               dist_peak_lesion=np.linalg.norm(pk - ly) / S,
               dist_peak_meanlesion=np.linalg.norm(pk - mean_les_c) / S,
               dist_peak_braincenter=np.linalg.norm(pk - bc) / S,
               peak_depth=float(dep[iy, ix]),
               top_frac_in_lesion=float((top & msk).sum() / max(top.sum(), 1)),
               top_frac_in_brain=float((top & brain).sum() / max(top.sum(), 1)),
               top_area=float(top.mean()),
               peak_in_lesion=bool(msk[iy, ix]))
    return out

def reference_values(msk, brain, mean_les_c):
    """Reference values without a model: a randomly drawn brain pixel."""
    S = msk.shape[0]; ly = centroid(msk)
    ys, xs = np.nonzero(brain)
    pts = np.c_[ys, xs].astype(float)
    dep = depth_map(brain)
    return dict(ref_dist_random_lesion=float(np.linalg.norm(pts - ly, axis=1).mean() / S),
                ref_depth_random=float(dep[brain].mean()),
                ref_depth_lesion=float(dep[int(round(ly[0])), int(round(ly[1]))]),
                dist_lesion_meanlesion=float(np.linalg.norm(ly - mean_les_c) / S),
                dist_lesion_braincenter=float(np.linalg.norm(ly - centroid(brain)) / S))

def main():
    import config as C
    from _xai_common import load_inputs, iter_models, roles_for
    import _xai_common as X
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="."); ap.add_argument("--cls", default="pituitary")
    ap.add_argument("--roles", default="deep", choices=["deep", "all"])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    if a.roles == "all": X.PRIMARY_ROLES = None
    xai, IMGS, MSK, LBL, CIX, BRAIN = load_inputs()
    idx = np.where(LBL == a.cls)[0]; print(f"{a.cls}: {len(idx)} images")
    I, M, B, Y = IMGS[idx], MSK[idx], BRAIN[idx], CIX[idx]; stems = xai.stem.values[idx]
    mean_les_c = np.nanmean([centroid(m) for m in M], axis=0)
    ref = [reference_values(M[i], B[i], mean_les_c) for i in range(len(idx))]
    print("mean lesion centroid (y,x):", mean_les_c.round(1))
    rows, acc, cnt = [], {}, {}
    for arch, seed, rid, model in iter_models():
        for role, layer in roles_for(arch):
            raw, post, up = X.cam_continuous(model, layer, I, Y)
            pmax = post.max(axis=(1, 2))
            key = (arch, role); acc.setdefault(key, np.zeros(up.shape[1:], float)); cnt.setdefault(key, 0)
            for i in range(len(idx)):
                d = describe(up[i], pmax[i], M[i], B[i], mean_les_c)
                d.update(ref[i], arch=arch, seed=seed, role=role, layer=layer, stem=stems[i])
                rows.append(d)
                if not d["degenerate"]: acc[key] += up[i]; cnt[key] += 1
        pd.DataFrame(rows).to_csv(f"{a.out}/B4b_per_image.csv", index=False)
        print(f"  {rid} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    d = pd.DataFrame(rows); d.to_csv(f"{a.out}/B4b_per_image.csv", index=False)
    nd = d[~d.degenerate]
    g = nd.groupby(["arch", "role"])
    s = g.agg(n_obs=("stem", "size"), dist_peak_lesion_med=("dist_peak_lesion", "median"),
              dist_peak_lesion_q1=("dist_peak_lesion", lambda v: v.quantile(.25)),
              dist_peak_lesion_q3=("dist_peak_lesion", lambda v: v.quantile(.75)),
              ref_dist_random_lesion_mean=("ref_dist_random_lesion", "mean"),
              dist_peak_meanlesion_med=("dist_peak_meanlesion", "median"),
              dist_lesion_meanlesion_med=("dist_lesion_meanlesion", "median"),
              frac_peak_closer_to_meanlesion=("dist_peak_meanlesion", lambda v: np.nan),   # filled in below
              peak_in_lesion_frac=("peak_in_lesion", "mean"),
              top_frac_in_lesion_mean=("top_frac_in_lesion", "mean"),
              top_frac_in_brain_mean=("top_frac_in_brain", "mean"),
              peak_depth_med=("peak_depth", "median"), ref_depth_lesion_med=("ref_depth_lesion", "median"),
              ref_depth_random_mean=("ref_depth_random", "mean"),
              frac_peak_rim=("peak_depth", lambda v: (v < 0.2).mean()),
              top_area_mean=("top_area", "mean"),
              peak_y_sd=("peak_y", "std"), peak_x_sd=("peak_x", "std"))
    nd2 = nd.assign(closer=nd.dist_peak_meanlesion < nd.dist_peak_lesion)
    s["frac_peak_closer_to_meanlesion"] = nd2.groupby(["arch", "role"]).closer.mean()
    s["frac_degenerate"] = d.groupby(["arch", "role"]).degenerate.mean()
    s = s.reset_index().round(4); s.to_csv(f"{a.out}/B4b_summary.csv", index=False)
    mean_maps = {f"{k[0]}__{k[1]}": acc[k] / max(cnt[k], 1) for k in acc}
    les_freq = M.mean(0); brain_freq = B.mean(0)
    np.savez_compressed(f"{a.out}/B4b_mean_maps.npz", lesion_freq=les_freq, brain_freq=brain_freq, **mean_maps)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = sorted(mean_maps); fig, ax = plt.subplots(1, len(keys) + 1, figsize=(3 * (len(keys) + 1), 3.2))
    ax[0].imshow(les_freq, cmap="magma"); ax[0].set_title("lesion frequency\n(%d images)" % len(idx), fontsize=8)
    for j, k in enumerate(keys, 1):
        ax[j].imshow(mean_maps[k], cmap="jet", vmin=0, vmax=1)
        ax[j].contour(les_freq, levels=[0.1], colors="w", linewidths=0.8)
        ax[j].set_title(k.replace("__", "\n"), fontsize=8)
    for x_ in ax: x_.axis("off")
    plt.tight_layout(); plt.savefig(f"{a.out}/B4b_mean_maps.png", dpi=150)
    print(s.T.to_string()); print("wrote B4b_* in", a.out)

if __name__ == "__main__": main()
