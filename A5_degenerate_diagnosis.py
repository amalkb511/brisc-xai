#!/usr/bin/env python3
"""
A5 (v2, adapted to this pipeline) — Diagnosis of the degenerate Grad-CAM maps of CNN_GAP.

Why a v2: the model ends with Dense(softmax); v1 differentiated the softmax output (probability),
whereas the manuscript uses the pre-softmax LOGIT. Here we go through models.CAM (linear head), hence
exactly the production path (03_measure_xai.py), and we read the gradients/activations more closely.

For each tumour image of the test set, for each CNN_GAP model (all seeds), layer 'deep' (28x28):
  degenerate        : post.max <= 1e-12 (same criterion as in production)
  raw_max, raw_min  : CAM before ReLU
  frac_raw_pos      : fraction of 28x28 pixels with CAM > 0
  frac_alpha_neg    : fraction of channels whose weight alpha = mean(dz/dA) is <= 0
  frac_alpha_zero   : same, alpha == 0 exactly
  grad_l2, frac_grad_zero : gradient norm, share of exactly-zero gradients
  act_mean, act_max, frac_ch_active : activation distribution of the target layer
  pos_mass, neg_mass: sum of positive / negative alpha*A contributions (max over space)
  logit_cls, p_cls, p_max : logit of the target class, probability, confidence
Reading: degenerate with grad_l2 ~ 0 / frac_grad_zero ~ 1  -> gradient cancellation (dead channels/ReLU);
          degenerate with grad_l2 > 0 and raw_max <= 0       -> entirely negative CAM (cancellation by ReLU, structural);
          high p_cls + large logit -> softmax saturation is NOT the cause (the target is the logit).
OUTPUT: --out (per image) and --out-summary (per seed x class).
Run :  python -u A5_degenerate_diagnosis.py --out A5_per_image.csv --out-summary A5_summary.csv
"""
import argparse, os, time
import numpy as np, pandas as pd, tensorflow as tf


def diagnose(cam, IMGS, CIX, batch=16):
    rows = []
    for i in range(0, len(IMGS), batch):
        x = tf.convert_to_tensor(IMGS[i:i + batch]); ci = tf.convert_to_tensor(CIX[i:i + batch], dtype=tf.int32)
        with tf.GradientTape() as tape:
            conv, out = cam.gm(x, training=False)
            tape.watch(conv)
            logits = tf.cast(out, tf.float32)
            score = tf.gather(logits, ci, axis=1, batch_dims=1)
        g = tape.gradient(score, conv)
        conv = tf.cast(conv, tf.float32); g = tf.cast(g, tf.float32)
        alpha = tf.reduce_mean(g, axis=(1, 2))                       # b x K
        contrib = tf.einsum("bhwc,bc->bhwc", conv, alpha)            # per-channel contributions
        raw = tf.reduce_sum(contrib, axis=-1).numpy()                 # b x h x w (pre-ReLU)
        post_max = np.maximum(raw, 0).reshape(len(raw), -1).max(1)
        conv_n, g_n, alpha_n, contrib_n = conv.numpy(), g.numpy(), alpha.numpy(), contrib.numpy()
        probs = tf.nn.softmax(logits).numpy(); lg = logits.numpy()
        for b in range(len(raw)):
            c = int(CIX[i + b])
            rows.append(dict(
                degenerate=bool(post_max[b] <= 1e-12),
                raw_max=float(raw[b].max()), raw_min=float(raw[b].min()),
                frac_raw_pos=float((raw[b] > 0).mean()),
                frac_alpha_neg=float((alpha_n[b] <= 0).mean()), frac_alpha_zero=float((alpha_n[b] == 0).mean()),
                grad_l2=float(np.linalg.norm(g_n[b])), frac_grad_zero=float((g_n[b] == 0).mean()),
                act_mean=float(conv_n[b].mean()), act_max=float(conv_n[b].max()),
                frac_ch_active=float((conv_n[b].max(axis=(0, 1)) > 0).mean()),
                pos_mass=float(np.maximum(contrib_n[b], 0).sum(-1).max()),
                neg_mass=float(np.minimum(contrib_n[b], 0).sum(-1).min()),
                logit_cls=float(lg[b, c]), p_cls=float(probs[b, c]), p_max=float(probs[b].max())))
    return rows


def main():
    import config as C
    from models import CAM
    from _xai_common import load_inputs, iter_models, roles_for
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="A5_per_image.csv"); ap.add_argument("--out-summary", default="A5_summary.csv")
    ap.add_argument("--arch", default="CNN_GAP")
    a = ap.parse_args(); t0 = time.time()
    xai, IMGS, MSK, LBL, CIX, _ = load_inputs()
    allrows = []
    for arch, seed, rid, model in iter_models():
        if arch != a.arch: continue
        for role, layer in roles_for(arch):
            cam = CAM(model, layer)
            try:
                err = cam.verify(IMGS)
                for k, r in enumerate(diagnose(cam, IMGS, CIX)):
                    r.update(arch=arch, seed=seed, role=role, layer=layer, stem=xai.stem.iloc[k],
                             true_class=LBL[k], verify_err=err)
                    allrows.append(r)
            finally:
                cam.close()
        pd.DataFrame(allrows).to_csv(a.out, index=False)
        print("  %s done (%.1f min)" % (rid, (time.time() - t0) / 60), flush=True)
    d = pd.DataFrame(allrows); d.to_csv(a.out, index=False)
    num = ["raw_max", "frac_raw_pos", "frac_alpha_neg", "grad_l2", "frac_grad_zero", "act_mean", "act_max",
           "frac_ch_active", "logit_cls", "p_cls"]
    s = (d.groupby(["seed", "true_class", "degenerate"]).agg(n=("stem", "size"), **{c: (c, "median") for c in num})
           .reset_index())
    s.to_csv(a.out_summary, index=False)
    print(d.groupby(["seed", "true_class"]).degenerate.agg(["sum", "size"]).rename(columns={"sum": "n_degenerate", "size": "n"}).to_string())
    print("wrote", a.out, a.out_summary, flush=True)


if __name__ == "__main__":
    main()
