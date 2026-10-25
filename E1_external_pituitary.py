"""
E1 - External check on pituitary adenomas (response to reviewer comment A9).

Applies the twenty frozen BRISC-trained models (4 architectures x 5 seeds, NO retraining)
to 2D slices cut from an independent dataset of pituitary adenomas with expert tumour
masks (Pandit et al., Scientific Data 2025, figshare 10.6084/m9.figshare.27894084), and
measures the same quantities as 03_measure_xai.py at the primary layer (28x28):
intra-brain AUC, energy lift, Dice, uniform floor, ceiling and the within-set
permuted-mask null. The Grad-CAM path is the one of 03_measure_xai.py (pre-softmax logit,
float32, ReLU, max-normalisation, threshold 0.5); the helper functions below are copies of
those of 03_measure_xai.py so that the numbers are directly comparable.

Brain mask: the Otsu mask of 03 fails on this set (see head_mask_ext); a low-threshold + hull mask is used instead, the Otsu
inside-fraction is stored per slice (inside_orig_mask). This is a documented deviation.

Slice preparation is done by the notebook Step1_external_pituitary.ipynb (T1 + tumour
files only; 224x224 PNG; mask PNG). Selection rules are fixed here, before any model is run:
  * exclude patients whose image and mask grids differ (2, 8, 9, 14, 16, 21, 73, 87) and
    patient 80 (anisotropic 1.5 mm voxels);
  * exclude slices within Hamming distance 4 of any BRISC image (pHash audit);
  * keep slices whose lesion covers >= MIN_FRAC of the frame (primary 0.2%; 0.5% = sensitivity).

Usage (Colab, GPU):
  python E1_external_pituitary.py --slices /content/slices --meta slices_meta_with_audit.csv \
         --models /content/models --out /content/E1_out
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import config as C
from models import CAM, assert_geometry, build_cnn

S = C.PROTOCOL["img_size"]
PIT = C.CLASS_NAMES.index("pituitary")          # class index of the true class (= 3)
EXCLUDED_PIDS = [2, 8, 9, 14, 16, 21, 73, 87, 80]
NEAR_DUP_MAX = 4
ROLE = "deep"


# ---- identical to 03_measure_xai.py -------------------------------------------------
def brain_mask(img):
    g = (img.mean(axis=2) * 255).astype(np.uint8)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return bw > 0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def head_mask_ext(img):
    """Foreground mask used for the EXTERNAL set only (deviation from 03_measure_xai.py).
    The Otsu mask of 03 keeps only the bright scalp/skull rim on many mid-sagittal T1c slices
    (it then contains ~0% of the lesion). Here: fixed low threshold (0.05 on the mean grey
    level, i.e. any non-background pixel), 7x7 closing, largest connected component, convex-hull
    fill. Same role (foreground/background split for the intra-brain AUC and the energy lift)."""
    g = (img.mean(axis=2) * 255).astype(np.uint8)
    bw = (g > int(0.05 * 255)).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return bw > 0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(comp)
    cv2.fillConvexPoly(out, cv2.convexHull(np.vstack(cnts)), 1)
    return out > 0


def auc_of(score, mask):
    m = mask.ravel()
    if m.min() == m.max():
        return np.nan
    return float(roc_auc_score(m, score.ravel()))


def uniform_hard_dice(p):
    return 2.0 * p / (1.0 + p)


def permuted_null(bin_maps, masks, rng, n_perm=None):
    """Each binarised map is scored against ANOTHER image's mask (single class here, so the
    permutation is over the whole external set), no fixed points, n_perm repeats."""
    n_perm = n_perm or C.NULL_REPS
    idx = np.arange(len(masks))
    acc = np.zeros(len(masks))
    for _ in range(n_perm):
        perm = rng.permutation(idx)
        for k in range(len(perm)):
            if perm[k] == idx[k]:
                j = (k + 1) % len(perm)
                perm[k], perm[j] = perm[j], perm[k]
        out = np.zeros(len(masks))
        for k, i in enumerate(idx):
            a_, b_ = bin_maps[i], masks[perm[k]]
            den = a_.sum() + b_.sum()
            out[i] = 2.0 * np.logical_and(a_, b_).sum() / den if den else 0.0
        acc += out
    return acc / n_perm


def controls(msk, brains):
    rng = np.random.default_rng(C.ANALYSIS_SEED)
    rows = []
    for name in ("perfect_mask", "gaussian_on_lesion", "uniform_random", "anti_lesion"):
        ai = []
        for i in range(min(len(msk), 200)):
            m, b = msk[i], brains[i]
            if name == "perfect_mask":
                s = m.astype("float32")
            elif name == "gaussian_on_lesion":
                ys, xs = np.nonzero(m)
                if not len(ys):
                    continue
                yy, xx = np.mgrid[0:S, 0:S]
                s = np.exp(-(((yy - ys.mean()) ** 2 + (xx - xs.mean()) ** 2)
                             / (2 * (S / 12) ** 2))).astype("float32")
            elif name == "uniform_random":
                s = rng.random((S, S)).astype("float32")
            else:
                s = (~m).astype("float32")
            ai.append(auc_of(s[b], m[b]))
        rows.append(dict(control=name, auc_intrabrain=np.nanmean(ai)))
    return pd.DataFrame(rows)
# --------------------------------------------------------------------------------------


BRISC_P99 = 0.6962       # mean p99 of the non-zero pixels of the 300 BRISC pituitary test images (E3 control)
BRISC_JPEG_Q = 75        # modal estimated JPEG quality of BRISC pituitary test images (233/300; 67/300 at 92)


def brisc_matched(imgs):
    """Present the external slices to the models the way BRISC images are presented, as far as it can be
    measured (BRISC documents neither its windowing nor its resizing): (i) per-slice linear rescaling so that the
    99th percentile of the non-zero pixels equals the BRISC mean (0.6962); (ii) JPEG re-encoding at the modal
    BRISC quality (75). Rule fixed from BRISC statistics only, before the matched analysis was run."""
    out = np.empty_like(imgs)
    for i, im in enumerate(imgs):
        g = im[..., 0]
        v = g[g > 0.05]
        hi = np.percentile(v, 99) if v.size else 1.0
        g = np.clip(g * (BRISC_P99 / max(hi, 1e-3)), 0, 1)
        ok, buf = cv2.imencode(".jpg", (g * 255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, BRISC_JPEG_Q])
        d = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
        out[i] = np.repeat(d[..., None], 3, axis=2)
    return out


def select_slices(meta, min_frac):
    m = meta.copy()
    m["pid"] = m.pid.astype(int)
    n0, p0 = len(m), m.pid.nunique()
    m = m[~m.pid.isin(EXCLUDED_PIDS)]
    n1 = len(m)
    m = m[m.d_brisc_all > NEAR_DUP_MAX]
    n2 = len(m)
    m = m[m.tumour_frac_224 >= min_frac].reset_index(drop=True)
    print(f"selection: {n0} slices/{p0} patients -> {n1} after patient exclusions -> {n2} after "
          f"near-duplicate removal -> {len(m)} slices/{m.pid.nunique()} patients with lesion >= {min_frac:.2%}")
    return m


def load_arrays(sel, slices_dir):
    IMGS, MSK = [], []
    for nm in sel.name:
        im = cv2.imread(os.path.join(slices_dir, "img", nm + ".png"), cv2.IMREAD_GRAYSCALE)
        mk = cv2.imread(os.path.join(slices_dir, "msk", nm + ".png"), cv2.IMREAD_GRAYSCALE)
        assert im is not None and mk is not None, nm
        im = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        IMGS.append(np.repeat(im[..., None], 3, axis=2))
        MSK.append((cv2.resize(mk, (S, S), interpolation=cv2.INTER_NEAREST) / 255.0) >= 0.5)
    return np.stack(IMGS), np.stack(MSK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--models", required=True, help="folder with {arch}__seed{seed}.keras")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-frac", type=float, default=0.002)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--preproc", choices=["raw", "brisc_matched"], default="raw",
                    help="raw: slices as extracted (pre-specified analysis); brisc_matched: intensity and JPEG matched to BRISC")
    ap.add_argument("--archs", default=",".join(C.ARCHS))
    ap.add_argument("--seeds", default=",".join(map(str, C.PROTOCOL["seeds"])))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    assert_geometry()

    sel = select_slices(pd.read_csv(a.meta), a.min_frac)
    sel.to_csv(os.path.join(a.out, f"E1_selection_{a.tag}.csv"), index=False)
    IMGS, MSK = load_arrays(sel, a.slices)
    ORIG = np.stack([brain_mask(im) for im in IMGS])           # mask of 03 (kept for the report)
    inside_orig = np.array([MSK[i][ORIG[i]].sum() / max(MSK[i].sum(), 1) for i in range(len(MSK))])
    print("original Otsu mask of 03: lesion pixels inside mean %.3f, %d/%d slices < 0.95 (NOT used)"
          % (inside_orig.mean(), (inside_orig < 0.95).sum(), len(inside_orig)))
    BRAIN = np.stack([head_mask_ext(im) for im in IMGS])
    inside = np.array([MSK[i][BRAIN[i]].sum() / max(MSK[i].sum(), 1) for i in range(len(MSK))])
    bg = 1.0 - BRAIN.reshape(len(BRAIN), -1).mean(axis=1)
    print("background fraction mean %.3f | lesion pixels inside brain mask: mean %.3f, min %.3f"
          % (bg.mean(), inside.mean(), inside.min()))
    assert inside.mean() >= 0.99, "brain mask loses >5% of lesion pixels - check before continuing"
    ctl = controls(MSK, BRAIN)
    print("instrument controls:\n", ctl.round(4).to_string(index=False))
    assert abs(ctl.set_index("control").loc["perfect_mask", "auc_intrabrain"] - 1.0) < 1e-6
    assert abs(ctl.set_index("control").loc["uniform_random", "auc_intrabrain"] - 0.5) < 0.02
    ctl.to_csv(os.path.join(a.out, f"E1_controls_{a.tag}.csv"), index=False)

    if a.preproc == "brisc_matched":      # model inputs only: masks and brain masks stay those of the raw slices
        IMGS = brisc_matched(IMGS)
        g = IMGS.mean(axis=3)
        print("matched slices: mean non-zero %.3f, mean p99 %.3f (BRISC pituitary: 0.281 / 0.696)"
              % (np.mean([x[x > 0.05].mean() for x in g]), np.mean([np.percentile(x[x > 0.05], 99) for x in g])))
    CIX = np.full(len(IMGS), PIT, dtype="int32")
    rows = []
    for arch in a.archs.split(","):
        for seed in map(int, a.seeds.split(",")):
            rid = f"{arch}__seed{seed}"
            w = os.path.join(a.models, rid + ".keras")
            if not os.path.exists(w):
                print("  missing", w)
                continue
            tf.keras.mixed_precision.set_global_policy("float32")
            tf.keras.backend.clear_session()
            model = build_cnn(**C.ARCHS[arch], name=arch)
            model.load_weights(w)
            probs = np.asarray(model.predict(IMGS, batch_size=32, verbose=0), dtype="float32")
            layer = C.LAYERS[arch][ROLE]
            cam = CAM(model, layer)
            try:
                cam.verify(IMGS)
                raw = cam.raw(IMGS, CIX)
            finally:
                cam.close()
            h, wd = raw.shape[1:3]
            post = np.maximum(raw, 0.0)
            up = tf.image.resize((post / (post.max(axis=(1, 2), keepdims=True) + 1e-8))[..., None],
                                 (S, S)).numpy()[..., 0]
            binm_all = up >= C.THRESHOLD
            null_d = permuted_null(binm_all, MSK, np.random.default_rng(C.ANALYSIS_SEED))
            for i in range(len(IMGS)):
                m, b = MSK[i], BRAIN[i]
                les_small = cv2.resize(m.astype("float32"), (wd, h), interpolation=cv2.INTER_AREA)
                brain_small = cv2.resize(b.astype("float32"), (wd, h), interpolation=cv2.INTER_AREA) > 0.5
                p, pb = post[i], post[i][brain_small]
                lb = les_small[brain_small]
                binm = binm_all[i]
                inter = float(np.logical_and(binm, m).sum())
                H, M = float(binm.mean()), float(m.mean())
                les_in_brain = (les_small > 0.5) & brain_small
                e_les = float(p[les_in_brain].sum()) if les_in_brain.any() else 0.0
                den = binm.sum() + m.sum()
                dice = 2 * inter / den if den else 0.0
                rows.append(dict(
                    arch=arch, seed=seed, run_id=rid, role=ROLE, layer=layer, resolution=f"{h}x{wd}",
                    inside_orig_mask=float(inside_orig[i]), name=sel.name.iloc[i], pid=int(sel.pid.iloc[i]), plane=sel.plane.iloc[i],
                    auc_intrabrain=auc_of(pb, lb > 0.5),
                    auc_full=auc_of(p, cv2.resize(m.astype(np.uint8), (wd, h),
                                                  interpolation=cv2.INTER_NEAREST) > 0),
                    energy_lift_brain=(e_les / (pb.sum() + 1e-12)) / max(float(lb.mean()), 1e-9),
                    dice=dice, coverage=inter / max(m.sum(), 1), precision=inter / max(binm.sum(), 1),
                    heatmap_area=H, lesion_area=M,
                    ceiling=2 * min(H, M) / (H + M) if (H + M) > 0 else np.nan,
                    null_dice=float(null_d[i]), dice_minus_null=dice - float(null_d[i]),
                    uniform_floor=float(uniform_hard_dice(M)),
                    degenerate=bool(p.max() <= 1e-12),
                    p_pituitary=float(probs[i, PIT]), pred=int(np.argmax(probs[i])),
                    correct=bool(np.argmax(probs[i]) == PIT)))
            r = pd.DataFrame(rows)
            r = r[r.run_id == rid]
            print(f"  {rid}: recall(pituitary)={r.correct.mean():.3f}  dice={r.dice.mean():.4f}  "
                  f"null={r.null_dice.mean():.4f}  auc_ib={r.auc_intrabrain.mean():.3f}  "
                  f"degenerate={r.degenerate.mean():.3f}")
            del model
    out = C.stamp(pd.DataFrame(rows))
    path = os.path.join(a.out, f"E1_per_image_{a.tag}.csv")
    out.to_csv(path, index=False)
    print(f"\n{len(out)} rows -> {path}")


if __name__ == "__main__":
    main()
