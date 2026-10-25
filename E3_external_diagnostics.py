"""
E3 - Diagnostics of the external pituitary check (E1).

Two questions, answered before any external number is interpreted:

  control   Does the loading / prediction code reproduce the BRISC results? The twenty frozen
            models are applied to the 1,000 BRISC test images with exactly the preprocessing of
            03_measure_xai.py (keras load_img, 224x224, /255). Expected: mean accuracy per
            architecture close to Table 3 (97.66 / 95.72 / 92.26 / 93.28 %) and pituitary
            sensitivity 0.986-0.996 (Table 6). Also prints simple intensity statistics of BRISC
            pituitary images next to those of the external slices.

  variants  Is the collapse of pituitary sensitivity on the external slices (0.20-0.30) due to
            a preprocessing mismatch (orientation, intensity) rather than to the domain? The same
            external slices are predicted under a few fixed transformations (flips, rotation,
            per-slice intensity stretch, gamma). A variant that restores sensitivity would mean
            that the external slices were not presented to the models like BRISC images.
            Diagnostic only: nothing here replaces the pre-specified analysis.

Usage (Colab, GPU):
  python E3_external_diagnostics.py control  --brisc-root /content/brisc_fresh --models /content/models \
         --slices /content/slices --meta /content/slices_meta_with_audit.csv --out /content/E3_out
  python E3_external_diagnostics.py variants --models /content/models --slices /content/slices \
         --meta /content/slices_meta_with_audit.csv --out /content/E3_out
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

import config as C
from models import build_cnn
from E1_external_pituitary import load_arrays, select_slices

S = C.PROTOCOL["img_size"]
PIT = C.CLASS_NAMES.index("pituitary")
TABLE3 = {"Baseline": 97.66, "CNN_GAP": 95.72, "CNN_GAP_BN": 92.26, "Proposed": 93.28}


def iter_models(models_dir, archs=None, seeds=None, policy="float32"):
    archs = archs or list(C.ARCHS)
    seeds = seeds or C.PROTOCOL["seeds"]
    for arch in archs:
        for seed in seeds:
            w = os.path.join(models_dir, f"{arch}__seed{seed}.keras")
            if not os.path.exists(w):
                print("  missing", w)
                continue
            tf.keras.mixed_precision.set_global_policy(policy)
            tf.keras.backend.clear_session()
            m = build_cnn(**C.ARCHS[arch], name=arch)
            m.load_weights(w)
            yield arch, seed, m


STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def jpeg_quality(path):
    """Estimated libjpeg quality from the first (luminance) quantisation table (sum-matched)."""
    from PIL import Image
    try:
        t = np.array(list(Image.open(path).quantization[0]))
    except Exception:
        return np.nan
    best, bq = 1e9, np.nan
    for q in range(1, 101):
        sc = 5000 / q if q < 50 else 200 - 2 * q
        tab = np.clip(np.floor((STD_LUMA * sc + 50) / 100), 1, 255)
        d = abs(tab.sum() - t.sum())
        if d < best:
            best, bq = d, q
    return bq


def load_one(path, loader):
    if loader == "tf":      # as in 02_train_ablation.py: decode_jpeg(channels=3) + tf.image.resize (bilinear)
        img = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(path), channels=3), [S, S])
        return (tf.cast(img, tf.float32) / 255.0).numpy()
    # as in 03_measure_xai.py: keras load_img (PIL, default 'nearest') + /255
    return tf.keras.utils.img_to_array(tf.keras.utils.load_img(path, target_size=(S, S))).astype("float32") / 255.0


def load_brisc_test(root, loader="keras"):
    cls_root = glob.glob(os.path.join(root, "**", "classification_task"), recursive=True)
    assert cls_root, f"classification_task not found under {root}"
    rows = []
    for idx, cls in enumerate(C.CLASS_NAMES):
        d = os.path.join(cls_root[0], "test", cls)
        assert os.path.isdir(d), d
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                mm = re.search(r"_(ax|co|sa)_", f)
                rows.append(dict(path=os.path.join(d, f), label=cls, y=idx, plane=mm[1] if mm else "?"))
    df = pd.DataFrame(rows)
    from PIL import Image
    df["size"] = [Image.open(p).size for p in df.path]
    df["jpeg_q"] = [jpeg_quality(p) for p in df.path]
    X = np.stack([load_one(p, loader) for p in df.path])
    return df, X


def intensity_stats(X, name):
    g = X.mean(axis=3)
    nz = [x[x > 0.05] for x in g]
    return dict(set=name, n=len(X), mean_all=float(g.mean()),
                mean_nonzero=float(np.mean([v.mean() for v in nz if v.size])),
                p99_nonzero=float(np.mean([np.percentile(v, 99) for v in nz if v.size])),
                frac_dark=float((g < 0.05).mean()))


def cmd_control(a):
    os.makedirs(a.out, exist_ok=True)
    df, X = load_brisc_test(a.brisc_root, a.loader)
    print("BRISC test:", len(df), df.label.value_counts().to_dict(), "| loader:", a.loader, "| policy:", a.policy)
    print("native image sizes (w,h):", df["size"].value_counts().head(6).to_dict())
    print("estimated JPEG quality (all test images):", df.jpeg_q.value_counts().head(6).to_dict())
    print("pituitary only - sizes:", df[df.label == "pituitary"]["size"].value_counts().head(4).to_dict(),
          "| quality:", df[df.label == "pituitary"].jpeg_q.value_counts().head(4).to_dict())
    rows = []
    for arch, seed, m in iter_models(a.models, policy=a.policy):
        p = np.asarray(m.predict(X, batch_size=32, verbose=0), dtype="float32")
        pred = p.argmax(1)
        ok = pred == df.y.values
        pit = df.y.values == PIT
        r = dict(arch=arch, seed=seed, acc=ok.mean(), pit_recall=ok[pit].mean())
        for pl in ("ax", "co", "sa"):
            r[f"pit_recall_{pl}"] = ok[pit & (df.plane.values == pl)].mean()
        rows.append(r)
        print(f"  {arch}__seed{seed}: acc={r['acc']:.4f}  pituitary recall={r['pit_recall']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, f"E3_control_brisc_{a.loader}_{a.policy}_per_run.csv"), index=False)
    s = res.groupby("arch")[["acc", "pit_recall", "pit_recall_ax", "pit_recall_co", "pit_recall_sa"]].mean()
    s["acc_pct"] = 100 * s.acc
    s["table3_pct"] = pd.Series(TABLE3)
    print("\nControl BRISC (mean over seeds):\n", s.round(4).to_string())
    s.to_csv(os.path.join(a.out, f"E3_control_brisc_{a.loader}_{a.policy}_summary.csv"))

    sel = select_slices(pd.read_csv(a.meta), 0.002)
    Xe, _ = load_arrays(sel, a.slices)
    pitX = X[df.y.values == PIT]
    st = pd.DataFrame([intensity_stats(pitX, "BRISC pituitary test"), intensity_stats(Xe, "external slices")])
    print("\nIntensity statistics:\n", st.round(4).to_string(index=False))
    st.to_csv(os.path.join(a.out, "E3_intensity_stats.csv"), index=False)


BRISC_P99, EXT_P99 = 0.7034, 0.9565      # mean p99 of non-zero pixels: BRISC pituitary test / external (E3 control)
BRISC_MEAN, EXT_MEAN = 0.2820, 0.4091    # mean of non-zero pixels, same sets


def variants_intensity(X):
    def to_p99(x, target=BRISC_P99):
        out = np.empty_like(x)
        for i, im in enumerate(x):
            v = im[..., 0][im[..., 0] > 0.05]
            hi = np.percentile(v, 99) if v.size else 1.0
            out[i] = np.clip(im * (target / max(hi, 1e-3)), 0, 1)
        return out
    def jpeg(x, q=90):
        out = np.empty_like(x)
        for i, im in enumerate(x):
            g = (im[..., 0] * 255).astype(np.uint8)
            ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, q])
            d = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
            out[i] = np.repeat(d[..., None], 3, axis=2)
        return out
    return {
        "identity": X,
        "jpeg90": jpeg(X),                                   # BRISC images are JPEG; external slices are PNG
        "scale_p99_global": X * (BRISC_P99 / EXT_P99),      # x0.735, same factor for all slices
        "scale_mean_global": X * (BRISC_MEAN / EXT_MEAN),   # x0.689
        "p99_to_brisc_perslice": to_p99(X),                 # each slice rescaled to the BRISC mean p99
    }


def variants_of(X):
    def stretch(x):
        out = np.empty_like(x)
        for i, im in enumerate(x):
            g = im[..., 0]
            v = g[g > 0.02]
            hi = np.percentile(v, 99) if v.size else 1.0
            out[i] = np.clip(im / max(hi, 1e-3), 0, 1)
        return out
    return {
        "identity": X,
        "fliplr": X[:, :, ::-1, :],
        "flipud": X[:, ::-1, :, :],
        "rot180": X[:, ::-1, ::-1, :],
        "stretch99": stretch(X),
        "gamma0.7": np.power(X, 0.7),
        "gamma1.4": np.power(X, 1.4),
    }


def cmd_variants(a):
    os.makedirs(a.out, exist_ok=True)
    sel = select_slices(pd.read_csv(a.meta), 0.002)
    rng = np.random.default_rng(0)
    keep = []
    for pl in ("axial", "coronal", "sagittal"):
        ix = np.where(sel.plane.values == pl)[0]
        keep += list(rng.choice(ix, size=min(a.per_plane, len(ix)), replace=False))
    keep = np.array(sorted(keep))
    sel = sel.iloc[keep].reset_index(drop=True)
    X, _ = load_arrays(sel, a.slices)
    V = variants_intensity(X) if a.set == "intensity" else variants_of(X)
    rows = []
    for arch, seed, m in iter_models(a.models):
        for name, Xv in V.items():
            p = np.asarray(m.predict(np.ascontiguousarray(Xv), batch_size=32, verbose=0), dtype="float32")
            ok = p.argmax(1) == PIT
            for pl in ("axial", "coronal", "sagittal"):
                rows.append(dict(arch=arch, seed=seed, variant=name, plane=pl, recall=ok[sel.plane.values == pl].mean()))
            rows.append(dict(arch=arch, seed=seed, variant=name, plane="all", recall=ok.mean()))
        print(f"  {arch}__seed{seed} done")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, f"E3_variants_{a.set}_per_run.csv"), index=False)
    t = res.groupby(["arch", "plane", "variant"]).recall.mean().unstack("variant")
    pd.set_option("display.width", 220)
    print(f"\nPituitary sensitivity on {len(sel)} external slices, mean over seeds:\n", t.round(3).to_string())
    t.to_csv(os.path.join(a.out, f"E3_variants_{a.set}_summary.csv"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("control")
    c.add_argument("--brisc-root", required=True)
    v = sub.add_parser("variants")
    v.add_argument("--per-plane", type=int, default=200)
    v.add_argument("--set", choices=["orientation", "intensity"], default="orientation")
    c.add_argument("--loader", choices=["keras", "tf"], default="keras", help="keras: as 03_measure_xai.py; tf: as 02_train_ablation.py")
    c.add_argument("--policy", default="float32", help="float32 (as 03_measure_xai.py) or mixed_float16 (as at training)")
    for p in (c, v):
        p.add_argument("--models", required=True)
        p.add_argument("--slices", required=True)
        p.add_argument("--meta", required=True)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"control": cmd_control, "variants": cmd_variants}[a.cmd](a)


if __name__ == "__main__":
    main()
