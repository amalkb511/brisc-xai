"""
Step 3 — Measure attention-lesion alignment, every model, every seed.

No retraining: loads the twelve frozen checkpoints and computes, per image, per
architecture, per layer role:

    auc_full         AUC over the whole frame           <- reported only as the artefact
    auc_intrabrain   AUC restricted to brain pixels     <- primary
    auc_upsampled    AUC after resize, whole frame       <- compare to auc_full
    auc_upsampled_ib AUC after resize, brain only        <- compare to auc_intrabrain
    auc_pre_relu_ib  same, on the pre-ReLU CAM
    energy_lift_brain  CAM energy in the lesion / lesion area, intra-brain (chance = 1)
    spearman         CAM vs area-downsampled lesion fraction, intra-brain
    dice, coverage, precision            thresholded overlap
    ceiling      2*min(H,M)/(H+M)        the most Dice this map size could score
    uniform_floor 2p/(1+p)               what a map active everywhere would score
    null_dice                            within-class permuted-mask null, 5 repeats
    dice_minus_null                      the paired excess over that null

Why the comparison set matters. Roughly 46% of a BRISC frame is empty space
outside the skull, and it is not constant across classes: 52.3% for glioma
against 36.8% for pituitary. On full-frame AUC that difference alone hands
glioma up to +0.078 before any model behaviour is measured, which is enough to
manufacture a class ordering. Restricting the comparison set to intra-brain
pixels removes it. Cell "controls" below verifies the instrument on synthetic
maps with no model in them: a perfect map must score 1.0, a uniform random map
0.5, an anti-lesion map 0.0.

Known measurement caveats, reported rather than hidden:
  * Spearman is undefined when the intra-brain CAM is constant. The NaN rate
    reaches 31.5% for meningioma on CNN_GAP and 0% for glioma everywhere, so it
    carries a class- and architecture-dependent selection bias and is NOT used
    as the primary metric.
  * Native-resolution AUC rests on few positive cells: ~15/22/7 at 28x28 and
    ~3.8/5.6/1.7 at 14x14 for glioma/meningioma/pituitary. The 28x28 role is
    primary for that reason, fixed before looking at results.

Outputs (in XAI_DIR): xai_per_image.csv, brain_mask_check.csv, instrument_controls.csv
Run:  python 03_measure_xai.py
"""
import json
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import config as C
from models import CAM, assert_geometry, build_cnn

S = C.PROTOCOL["img_size"]


def brain_mask(img):
    """Otsu on the grayscale image, closed, largest connected component."""
    g = (img.mean(axis=2) * 255).astype(np.uint8)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return bw > 0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def auc_of(score, mask):
    m = mask.ravel()
    if m.min() == m.max():
        return np.nan
    return float(roc_auc_score(m, score.ravel()))


def uniform_hard_dice(p):
    """A spatially uninformative map that is active everywhere scores 2p/(1+p)
    against a mask of area fraction p. This is the structural floor of Dice: it
    depends only on lesion size, not on anything the model learned."""
    return 2.0 * p / (1.0 + p)


def permuted_null_within_class(bin_maps, masks, classes, rng, n_perm=None):
    """Score each image's binarised map against ANOTHER image's mask from the SAME class.

    Permuting within class holds lesion size and the stereotyped anatomical position
    of each tumour type constant. Permuting across the whole masked set instead
    inflates the pituitary null roughly twofold, because pituitary masks are the
    smallest in this dataset - which would make the smallest-lesion class appear
    further below chance than it is.
    """
    n_perm = n_perm or C.NULL_REPS
    classes = np.asarray(classes)
    acc = np.zeros(len(masks))
    for _ in range(n_perm):
        out = np.zeros(len(masks))
        for cls in np.unique(classes):
            idx = np.where(classes == cls)[0]
            perm = rng.permutation(idx)
            for k in range(len(perm)):          # forbid fixed points
                if perm[k] == idx[k]:
                    j = (k + 1) % len(perm)
                    perm[k], perm[j] = perm[j], perm[k]
            for k, i in enumerate(idx):
                a_, b_ = bin_maps[i], masks[perm[k]]
                den = a_.sum() + b_.sum()
                out[i] = 2.0 * np.logical_and(a_, b_).sum() / den if den else 0.0
        acc += out
    return acc / n_perm


def controls(msk, brains):
    """The instrument check. No model is involved."""
    rng = np.random.default_rng(C.ANALYSIS_SEED)
    rows = []
    for name in ("perfect_mask", "gaussian_on_lesion", "uniform_random", "anti_lesion"):
        af, ai = [], []
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
            af.append(auc_of(s, m))
            ai.append(auc_of(s[b], m[b]))
        rows.append(dict(control=name, auc_full=np.nanmean(af),
                         auc_intrabrain=np.nanmean(ai)))
    return pd.DataFrame(rows)


def main():
    C.ensure_dirs()
    assert_geometry()

    frozen = json.load(open(os.path.join(C.TRAIN_DIR, "frozen_protocol.json")))
    assert frozen["hash"] == C.PROTOCOL_HASH, (
        f"protocol hash mismatch: models on disk are {frozen['hash']}, config is "
        f"{C.PROTOCOL_HASH}. These are not the models you reported.")

    mani = pd.read_csv(os.path.join(C.DATASET_DIR, "manifest.csv"))
    xai = mani[(mani.split == "test") & mani.mask_path.notna()
               & mani.label.isin(C.TUMOR_CLASSES)].reset_index(drop=True)
    IMGS = np.stack([tf.keras.utils.img_to_array(
        tf.keras.utils.load_img(p, target_size=(S, S))).astype("float32") / 255.0
        for p in xai.path])
    MSK = np.stack([(cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (S, S),
                                interpolation=cv2.INTER_NEAREST) / 255.0) >= 0.5
                    for p in xai.mask_path])
    LBL, CIX = xai.label.values, xai.y.values.astype("int32")
    print(f"XAI set: {len(xai)} images | {pd.Series(LBL).value_counts().to_dict()}")

    BRAIN = np.stack([brain_mask(im) for im in IMGS])
    inside = np.array([MSK[i][BRAIN[i]].sum() / max(MSK[i].sum(), 1) for i in range(len(MSK))])
    bgfrac = 1.0 - BRAIN.reshape(len(BRAIN), -1).mean(axis=1)
    chk = pd.DataFrame(dict(true_class=LBL, bg_frac=bgfrac, lesion_inside_brain=inside))
    print("\nbackground fraction: mean %.3f | lesion pixels inside brain mask: mean %.3f"
          % (bgfrac.mean(), inside.mean()))
    print(chk.groupby("true_class")[["bg_frac", "lesion_inside_brain"]].mean().round(4))
    assert inside.mean() >= 0.95, "brain mask loses >5% of lesion pixels - fix it first"

    # A small brain mask is not an error: some slices are taken near the vertex or
    # the periphery, where the head occupies little of the frame. Report the count
    # as a limitation rather than aborting - their intra-brain metrics simply rest
    # on fewer pixels.
    frac = BRAIN.reshape(len(BRAIN), -1).mean(axis=1)
    chk["brain_frac"] = frac
    low = frac < C.BRAIN_MIN_FRAC
    if low.any():
        print(f"NOTE: {low.sum()}/{len(frac)} images have a brain mask below "
              f"{C.BRAIN_MIN_FRAC:.0%} of the frame (min {frac.min():.3f}).")
        print(chk.loc[low, ["true_class", "brain_frac"]].to_string())
    C.stamp(chk).to_csv(os.path.join(C.XAI_DIR, "brain_mask_check.csv"), index=False)

    ctl = controls(MSK, BRAIN)
    C.stamp(ctl).to_csv(os.path.join(C.XAI_DIR, "instrument_controls.csv"), index=False)
    print("\ninstrument controls:\n", ctl.round(4).to_string(index=False))
    assert abs(ctl.set_index("control").loc["perfect_mask", "auc_intrabrain"] - 1.0) < 1e-6
    assert abs(ctl.set_index("control").loc["uniform_random", "auc_intrabrain"] - 0.5) < 0.02

    runs = pd.read_csv(os.path.join(C.TRAIN_DIR, "ablation_runs.csv"))
    runs = runs[~runs.get("diverged", pd.Series(False, index=runs.index)).fillna(False).astype(bool)]

    rows = []
    for _, r in runs.iterrows():
        arch, seed, rid = r["arch"], int(r["seed"]), r["run_id"]
        w = os.path.join(C.TRAIN_DIR, "models", f"{rid}.keras")
        if arch not in C.LAYERS or not os.path.exists(w):
            print("  skip", rid)
            continue
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.backend.clear_session()
        model = build_cnn(**C.ARCHS[arch], name=arch)
        model.load_weights(w)

        for role, layer in C.LAYERS[arch].items():
            cam = CAM(model, layer)
            try:
                cam.verify(IMGS)
                raw = cam.raw(IMGS, CIX)
            finally:
                cam.close()
            h, wd = raw.shape[1:3]
            post = np.maximum(raw, 0.0)
            up = tf.image.resize(
                (post / (post.max(axis=(1, 2), keepdims=True) + 1e-8))[..., None],
                (S, S)).numpy()[..., 0]

            # Empirical null: same maps, mismatched masks, within class.
            binm_all = up >= C.THRESHOLD
            null_d = permuted_null_within_class(
                binm_all, MSK, LBL, np.random.default_rng(C.ANALYSIS_SEED))

            for i in range(len(IMGS)):
                m, b = MSK[i], BRAIN[i]
                les_small = cv2.resize(m.astype("float32"), (wd, h),
                                       interpolation=cv2.INTER_AREA)
                brain_small = cv2.resize(b.astype("float32"), (wd, h),
                                         interpolation=cv2.INTER_AREA) > 0.5
                p, pb = post[i], post[i][brain_small]
                lb = les_small[brain_small]
                with np.errstate(invalid="ignore"):
                    rho = (np.nan if pb.min() == pb.max() or lb.min() == lb.max()
                           else float(spearmanr(pb, lb).statistic))
                binm = binm_all[i]
                inter = float(np.logical_and(binm, m).sum())
                H, M = float(binm.mean()), float(m.mean())
                les_in_brain = (les_small > 0.5) & brain_small
                e_les = float(p[les_in_brain].sum()) if les_in_brain.any() else 0.0
                rows.append(dict(
                    arch=arch, seed=seed, run_id=rid, role=role, layer=layer,
                    resolution=f"{h}x{wd}", true_class=LBL[i], stem=xai.stem.iloc[i],
                    auc_full=auc_of(p, cv2.resize(m.astype(np.uint8), (wd, h),
                                                  interpolation=cv2.INTER_NEAREST) > 0),
                    auc_intrabrain=auc_of(pb, les_small[brain_small] > 0.5),
                    # Resolution effect, isolated. auc_upsampled must share the
                    # comparison set of auc_full (whole frame) so that the only
                    # difference between them is the interpolation: comparing it to
                    # auc_intrabrain instead would confound resolution with the
                    # choice of comparison set and yields a sign that flips between
                    # cells. Bilinear upsampling inflates AUC in every cell measured,
                    # and most for compact maps, so an upsampled between-class
                    # comparison partly measures map compactness. Recorded, not primary.
                    auc_upsampled=auc_of(up[i], m),
                    auc_upsampled_ib=auc_of(up[i][b], m[b]),
                    auc_pre_relu_ib=auc_of(raw[i][brain_small],
                                           les_small[brain_small] > 0.5),
                    spearman=rho,
                    energy_lift_brain=(e_les / (pb.sum() + 1e-12)) / max(
                        float(lb.mean()), 1e-9),
                    cam_mean_background=float(p[~brain_small].mean()),
                    cam_mean_brain_nonlesion=float(p[brain_small & (les_small <= 0.5)].mean()),
                    cam_mean_lesion=float(p[les_small > 0.5].mean())
                    if (les_small > 0.5).any() else np.nan,
                    background_fraction=float((~brain_small).mean()),
                    dice=2 * inter / (binm.sum() + m.sum()) if (binm.sum() + m.sum()) else 0.0,
                    coverage=inter / max(m.sum(), 1), precision=inter / max(binm.sum(), 1),
                    heatmap_area=H, lesion_area=M,
                    ceiling=2 * min(H, M) / (H + M) if (H + M) > 0 else np.nan,
                    null_dice=float(null_d[i]),
                    dice_minus_null=(2 * inter / (binm.sum() + m.sum())
                                     if (binm.sum() + m.sum()) else 0.0) - float(null_d[i]),
                    uniform_floor=float(uniform_hard_dice(M)),
                    degenerate=bool(p.max() <= 1e-12)))
        del model
        print(" ", rid, "done")

    out = C.stamp(pd.DataFrame(rows))
    path = os.path.join(C.XAI_DIR, "xai_per_image.csv")
    out.to_csv(path, index=False)
    print(f"\n{len(out)} rows -> {path}")
    print("\nNaN rate on spearman (why it is not the primary metric):")
    print(out.groupby(["arch", "role", "true_class"]).spearman
             .apply(lambda s: s.isna().mean()).round(3).to_string())


if __name__ == "__main__":
    main()
