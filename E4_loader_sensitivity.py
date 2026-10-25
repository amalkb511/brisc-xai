"""
E4 - Sensitivity of the BRISC saliency measurements to the image loader.

02_train_ablation.py reads the images with tf.image.decode_jpeg + tf.image.resize (bilinear); 03_measure_xai.py
(and _xai_common.py) read them with keras load_img, whose default interpolation is nearest neighbour. On the
512x512 BRISC images this is not the same 224x224 input: with the training loader the 20 models reproduce the
accuracies of Table 3 (E3 control), with the 03 loader some BatchNorm runs are 1-5 points lower. This script re-runs
03_measure_xai.py unchanged EXCEPT that load_img is replaced by the training loader, then compares the per-image
table with the published data_tables/xai_per_image.csv (same models, same images, same masks).

Usage (Colab, GPU; BRISC under /content/brisc_fresh/brisc2025, the 20 .keras files in /content/models):
  python E4_loader_sensitivity.py --models /content/models --out /content/E4_keras --loader keras      # reproduces the published table?
  python E4_loader_sensitivity.py --models /content/models --out /content/E4_training --loader training
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--loader", choices=["training", "keras"], default="training",
                help="training: tf.image.resize as in 02_train_ablation.py; keras: 03_measure_xai.py unchanged (reproducibility check)")
a = ap.parse_args()
out = os.path.abspath(a.out)
os.environ["BRISC_OUT"] = out
os.makedirs(os.path.join(out, "dataset"), exist_ok=True)
os.makedirs(os.path.join(out, "ablation", "models"), exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

import tensorflow as tf  # noqa: E402
import config as C  # noqa: E402

shutil.copy(os.path.join(here, "data_tables", "manifest.csv"), os.path.join(out, "dataset", "manifest.csv"))
json.dump({"hash": C.PROTOCOL_HASH}, open(os.path.join(out, "ablation", "frozen_protocol.json"), "w"))
runs = pd.read_csv(os.path.join(here, "results", "revision", "ablation_runs_main20.csv"))
runs.to_csv(os.path.join(out, "ablation", "ablation_runs.csv"), index=False)
for rid in runs.run_id:
    dst = os.path.join(out, "ablation", "models", f"{rid}.keras")
    if not os.path.exists(dst):
        shutil.copy(os.path.join(a.models, f"{rid}.keras"), dst)

_orig = tf.keras.utils.load_img


def load_img_training(path, target_size=None, **kw):       # as in 02_train_ablation.py
    img = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(path), channels=3), list(target_size))
    return img.numpy().astype("float32")


if a.loader == "training":
    tf.keras.utils.load_img = load_img_training
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("measure03", os.path.join(here, "03_measure_xai.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.main()

new = pd.read_csv(os.path.join(C.XAI_DIR, "xai_per_image.csv"))
old = pd.read_csv(os.path.join(here, "data_tables", "xai_per_image.csv"))
key = ["arch", "seed", "role", "true_class", "stem"]
j = old.merge(new, on=key, suffixes=("_pub", "_train"))
print(f"\nrows: published {len(old)}, re-run {len(new)}, matched {len(j)}")
mets = ["auc_intrabrain", "energy_lift_brain", "dice", "null_dice", "dice_minus_null", "ceiling", "uniform_floor", "degenerate"]
rows = []
for (arch, role, cls), g in j.groupby(["arch", "role", "true_class"]):
    r = dict(arch=arch, role=role, true_class=cls)
    for k in mets:
        r[k + "_pub"] = g[k + "_pub"].astype(float).mean()
        r[k + "_train"] = g[k + "_train"].astype(float).mean()
        r[k + "_diff"] = r[k + "_train"] - r[k + "_pub"]
    rows.append(r)
cmp_ = pd.DataFrame(rows)
cmp_.to_csv(os.path.join(out, "E4_loader_comparison.csv"), index=False)
pd.set_option("display.width", 250)
d = cmp_[cmp_.role == "deep"][["arch", "true_class"] + [k + s for k in ("auc_intrabrain", "dice", "dice_minus_null") for s in ("_pub", "_train", "_diff")]]
print("\nDeep layer (28x28), mean per architecture x class: published vs training loader\n", d.round(4).to_string(index=False))
pi = j[j.role == "deep"]
for k in ("auc_intrabrain", "dice", "dice_minus_null", "energy_lift_brain"):
    x = (pi[k + "_train"].astype(float) - pi[k + "_pub"].astype(float)).abs()
    print(f"{k}: per-image |difference| mean {x.mean():.4f}, 95th pct {x.quantile(.95):.4f}, max {x.max():.4f}")
