"""
Helpers shared by A6 and A7 — EXACTLY REPRODUCE the path of 03_measure_xai.py:
manifest/model loading, CAM.raw() -> ReLU -> upsample+max-norm -> threshold.
No .npy files are read: the maps are recomputed as in production.
"""
import json, os
import cv2, numpy as np, pandas as pd, tensorflow as tf
import config as C
from models import CAM, assert_geometry, build_cnn

S = C.PROTOCOL["img_size"]

# Role(s) to analyse. Primary = 28x28 ('deep'). Set to None for all roles of C.LAYERS.
PRIMARY_ROLES = ["deep"]


def brain_mask(img):
    """Identical to 03_measure_xai.brain_mask: Otsu, closing, largest component."""
    g = (img.mean(axis=2) * 255).astype(np.uint8)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return bw > 0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def load_inputs():
    """Reproduces 03_measure_xai.main() up to the loading of the images/masks (lines 102-119)."""
    C.ensure_dirs()
    assert_geometry()
    frozen = json.load(open(os.path.join(C.TRAIN_DIR, "frozen_protocol.json")))
    assert frozen["hash"] == C.PROTOCOL_HASH, (
        f"protocol hash mismatch: models on disk are {frozen['hash']}, config is {C.PROTOCOL_HASH}.")
    mani = pd.read_csv(os.path.join(C.DATASET_DIR, "manifest.csv"))
    xai = mani[(mani.split == "test") & mani.mask_path.notna()
               & mani.label.isin(C.TUMOR_CLASSES)].reset_index(drop=True)
    IMGS = np.stack([tf.keras.utils.img_to_array(
        tf.keras.utils.load_img(p, target_size=(S, S))).astype("float32") / 255.0
        for p in xai.path])
    MSK = np.stack([(cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (S, S),
                                interpolation=cv2.INTER_NEAREST) / 255.0) >= 0.5
                    for p in xai.mask_path])
    LBL = xai.label.values
    CIX = xai.y.values.astype("int32")
    BRAIN = np.stack([brain_mask(im) for im in IMGS])
    print(f"XAI set: {len(xai)} images | {pd.Series(LBL).value_counts().to_dict()}")
    return xai, IMGS, MSK, LBL, CIX, BRAIN


def iter_models():
    """Reproduces 03_measure_xai lines 136-149: each non-diverged run -> (arch, seed, run_id, model)."""
    runs = pd.read_csv(os.path.join(C.TRAIN_DIR, "ablation_runs.csv"))
    runs = runs[~runs.get("diverged", pd.Series(False, index=runs.index)).fillna(False).astype(bool)]
    for _, r in runs.iterrows():
        arch, seed, rid = r["arch"], int(r["seed"]), r["run_id"]
        w = os.path.join(C.TRAIN_DIR, "models", f"{rid}.keras")
        if arch not in C.LAYERS or not os.path.exists(w):
            print("  skip", rid); continue
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.backend.clear_session()
        model = build_cnn(**C.ARCHS[arch], name=arch)
        model.load_weights(w)
        yield arch, seed, rid, model
        del model


def roles_for(arch):
    items = C.LAYERS[arch].items()
    if PRIMARY_ROLES is None:
        return list(items)
    return [(r, l) for r, l in items if r in PRIMARY_ROLES]


def cam_continuous(model, layer, IMGS, CIX):
    """Reproduces 03_measure_xai lines 152-163:
       raw (pre-ReLU) -> post=max(raw,0) -> up = upsample(post / per-image max) in [0,1].
       Returns (raw, post, up). 'up' is the continuous full-frame max-normalised SxS map."""
    cam = CAM(model, layer)
    try:
        cam.verify(IMGS)
        raw = cam.raw(IMGS, CIX)
    finally:
        cam.close()
    post = np.maximum(raw, 0.0)
    up = tf.image.resize(
        (post / (post.max(axis=(1, 2), keepdims=True) + 1e-8))[..., None],
        (S, S)).numpy()[..., 0]
    return raw, post, up


def dice_bin(a, b):
    inter = float(np.logical_and(a, b).sum()); s = float(a.sum() + b.sum())
    return (2.0 * inter / s) if s > 0 else 0.0
