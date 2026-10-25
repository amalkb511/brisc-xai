"""
Step 5 — The Grad-CAM target x precision 2x2, on the frozen models.

Grad-CAM is defined on the pre-softmax class score, but implementations that
differentiate the post-softmax probability are common and the difference is
rarely stated. The two are not interchangeable for confidently predicted
classes: as p -> 1 the softmax Jacobian p(1-p) collapses, so the gradient that
reaches the target feature map is scaled by a vanishing factor. In reduced
precision it can underflow entirely, producing a CAM that is identically zero
after the ReLU.

This script measures the size of that effect, fully crossed:

    target     : post-softmax probability   vs   pre-softmax logit
    arithmetic : mixed_float16              vs   float32

on every frozen model and every masked test image, at the primary layer. It
reports, per architecture x class x cell:

    dice, auc_intrabrain      the alignment actually measured
    degenerate_frac           share of CAMs that are identically zero after ReLU
    mean_confidence           mean predicted probability of the true class

Nothing is retrained. ~20 min for 20 models.

Read the output as a 2x2: comparing the two rows isolates the precision effect,
comparing the two columns isolates the target effect, and the interaction tells
you whether precision matters once the target is correct.

Outputs (in XAI_DIR): gradcam_2x2.csv, gradcam_2x2_summary.csv
Run:  python 05_gradcam_target_2x2.py
"""
import json
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

import config as C
from models import build_cnn, assert_geometry
from importlib.machinery import SourceFileLoader

_m3 = SourceFileLoader("_m3", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "03_measure_xai.py")).load_module()
brain_mask, auc_of = _m3.brain_mask, _m3.auc_of

S = C.PROTOCOL["img_size"]


class CAM2x2:
    """Grad-CAM with the target switchable between the probability and the logit.

    tape.watch() takes the forward-pass tensor BEFORE any cast: tf.cast creates a
    new tensor that is off the backward path, so watching the cast copy makes
    tape.gradient return None under mixed_float16. The cast to float32 happens
    AFTER the backward pass, so that fp16 underflow stays measurable instead of
    being hidden by an early promotion.
    """

    def __init__(self, model, layer_name, use_logits):
        self.model, self.use_logits = model, use_logits
        self.pred = model.get_layer("predictions")
        self._act = self.pred.activation
        if use_logits:
            self.pred.activation = tf.keras.activations.linear
        self.gm = tf.keras.Model(model.inputs,
                                 [model.get_layer(layer_name).output, model.outputs[0]])

    def close(self):
        self.pred.activation = self._act

    def verify(self, x):
        if not self.use_logits:
            return 0.0
        _, out = self.gm(x[:8], training=False)
        got = tf.nn.softmax(tf.cast(out, tf.float32)).numpy()
        self.pred.activation = self._act
        ref = np.asarray(self.model(x[:8], training=False), dtype=np.float32)
        self.pred.activation = tf.keras.activations.linear
        err = float(np.abs(got - ref).max())
        assert err < 2e-2, f"logit head does not reproduce probabilities ({err:.2e})"
        return err

    @tf.function(reduce_retracing=True)
    def _cam(self, x, ci):
        with tf.GradientTape() as tape:
            conv, out = self.gm(x, training=False)
            tape.watch(conv)
            score = tf.gather(tf.cast(out, tf.float32), ci, axis=1, batch_dims=1)
        g = tape.gradient(score, conv)
        if g is None:
            raise RuntimeError("gradient is None: watched tensor is off the backward path")
        g = tf.cast(g, tf.float32)
        conv = tf.cast(conv, tf.float32)
        return tf.einsum("bhwc,bc->bhw", conv, tf.reduce_mean(g, axis=(1, 2)))

    def raw(self, imgs, cls_idx, batch=16):
        return np.concatenate([
            self._cam(tf.convert_to_tensor(imgs[i:i + batch]),
                      tf.convert_to_tensor(cls_idx[i:i + batch], dtype=tf.int32)).numpy()
            for i in range(0, len(imgs), batch)])


def main():
    C.ensure_dirs()
    assert_geometry()
    frozen = json.load(open(os.path.join(C.TRAIN_DIR, "frozen_protocol.json")))
    assert frozen["hash"] == C.PROTOCOL_HASH, (
        f"protocol hash mismatch: models on disk are {frozen['hash']}")

    mani = pd.read_csv(os.path.join(C.DATASET_DIR, "manifest.csv"))
    xai = mani[(mani.split == "test") & mani.mask_path.notna()
               & mani.label.isin(C.TUMOR_CLASSES)].reset_index(drop=True)
    IMGS = np.stack([tf.keras.utils.img_to_array(
        tf.keras.utils.load_img(p, target_size=(S, S))).astype("float32") / 255.0
        for p in xai.path])
    MSK = np.stack([(cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (S, S),
                                interpolation=cv2.INTER_NEAREST) / 255.0) >= 0.5
                    for p in xai.mask_path])
    BRAIN = np.stack([brain_mask(im) for im in IMGS])
    LBL, CIX = xai.label.values, xai.y.values.astype("int32")
    print(f"XAI set: {len(xai)} images")

    runs = pd.read_csv(os.path.join(C.TRAIN_DIR, "ablation_runs.csv"))
    runs = runs[~runs.get("diverged", pd.Series(False, index=runs.index))
                .fillna(False).astype(bool)]

    rows = []
    for _, r in runs.iterrows():
        arch, seed, rid = r["arch"], int(r["seed"]), r["run_id"]
        w = os.path.join(C.TRAIN_DIR, "models", f"{rid}.keras")
        if arch not in C.LAYERS or not os.path.exists(w):
            continue
        layer = C.LAYERS[arch][C.PRIMARY_ROLE]

        for policy in ("mixed_float16", "float32"):
            tf.keras.backend.clear_session()
            tf.keras.mixed_precision.set_global_policy(policy)
            model = build_cnn(**C.ARCHS[arch], name=arch)
            model.load_weights(w)
            # confidence in the true class, measured once per policy
            probs = np.concatenate([
                np.asarray(model(IMGS[i:i + 32], training=False), dtype=np.float32)
                for i in range(0, len(IMGS), 32)])
            conf = probs[np.arange(len(CIX)), CIX]

            for use_logits in (False, True):
                cam = CAM2x2(model, layer, use_logits)
                try:
                    cam.verify(IMGS)
                    raw = cam.raw(IMGS, CIX)
                finally:
                    cam.close()
                h, wd = raw.shape[1:3]
                post = np.maximum(raw, 0.0)
                peak = post.max(axis=(1, 2))
                up = tf.image.resize((post / (peak.reshape(-1, 1, 1) + 1e-8))[..., None],
                                     (S, S)).numpy()[..., 0]
                for i in range(len(IMGS)):
                    binm = up[i] >= C.THRESHOLD
                    inter = float(np.logical_and(binm, MSK[i]).sum())
                    den = binm.sum() + MSK[i].sum()
                    bs = cv2.resize(BRAIN[i].astype("float32"), (wd, h),
                                    interpolation=cv2.INTER_AREA) > 0.5
                    ls = cv2.resize(MSK[i].astype("float32"), (wd, h),
                                    interpolation=cv2.INTER_AREA)
                    rows.append(dict(
                        arch=arch, seed=seed, run_id=rid, layer=layer,
                        policy=policy,
                        target="logit" if use_logits else "softmax",
                        true_class=LBL[i], stem=xai.stem.iloc[i],
                        confidence=float(conf[i]),
                        dice=2 * inter / den if den else 0.0,
                        auc_intrabrain=auc_of(post[i][bs], ls[bs] > 0.5),
                        degenerate=bool(peak[i] <= 1e-12)))
            del model
        print(" ", rid, "done")

    out = C.stamp(pd.DataFrame(rows))
    out.to_csv(os.path.join(C.XAI_DIR, "gradcam_2x2.csv"), index=False)

    summ = (out.groupby(["target", "policy", "true_class"])
            .agg(dice=("dice", "mean"), auc=("auc_intrabrain", "mean"),
                 degenerate_frac=("degenerate", "mean"),
                 mean_confidence=("confidence", "mean"), n=("dice", "size"))
            .round(4))
    C.stamp(summ.reset_index()).to_csv(
        os.path.join(C.XAI_DIR, "gradcam_2x2_summary.csv"), index=False)

    print("\n" + "=" * 84)
    print("GRAD-CAM TARGET x PRECISION, primary layer, all frozen models")
    print("=" * 84)
    print(summ.to_string())
    print("\nDice, as a 2x2 per class (rows = precision, cols = target):")
    print(out.pivot_table(index=["true_class", "policy"], columns="target",
                          values="dice").round(4).to_string())
    print("\nDegenerate (all-zero) CAM fraction:")
    print(out.pivot_table(index=["true_class", "policy"], columns="target",
                          values="degenerate").round(4).to_string())
    print("\nMean confidence in the true class, by class:")
    print(out.groupby("true_class").confidence.mean().round(4).to_string())
    print("\n -> compare rows for the precision effect, columns for the target effect.")
    print("    A target effect concentrated in the highest-confidence class is the")
    print("    signature the softmax-saturation mechanism predicts.")


if __name__ == "__main__":
    main()
