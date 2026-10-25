#!/usr/bin/env python3
"""Figure 1: training and validation accuracy / loss for one seed of each architecture.
Reads the per-run histories written by 02_train_ablation.py (<BRISC_OUT>/ablation/histories/<arch>__seed<seed>.json).
Usage: BRISC_OUT=... python B7_fig1_training_curves.py [--seed 42] [--out <dir>]
"""
import argparse, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

RUNS = [("Baseline_CNN (Flatten)", "Baseline", "#1f4e8c"), ("CNN_GAP", "CNN_GAP", "#2e8b3a"),
        ("CNN_GAP_BN", "CNN_GAP_BN", "#c8372d"), ("CNN_GAP_BN_SE", "Proposed", "#8a4aa8")]  # "Proposed" is the run name in the files

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default=os.path.join(C.XAI_DIR, "figures")); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.7, 3.05), dpi=200)
for lab, arch, c in RUNS:
    h = json.load(open(os.path.join(C.TRAIN_DIR, "histories", f"{arch}__seed{a.seed}.json")))
    ep = range(1, len(h["accuracy"]) + 1)
    ax[0].plot(ep, h["accuracy"], c=c, lw=1.6, label=lab); ax[0].plot(ep, h["val_accuracy"], c=c, lw=0.9, ls="--", alpha=.8)
    ax[1].plot(ep, h["loss"], c=c, lw=1.6); ax[1].plot(ep, h["val_loss"], c=c, lw=0.9, ls="--", alpha=.8)
ax[0].set_title("(a) Training (solid) and validation (dashed) accuracy", fontsize=9); ax[0].set_ylabel("Accuracy")
ax[1].set_title("(b) Training (solid) and validation (dashed) loss", fontsize=9); ax[1].set_ylabel("Loss")
for x in ax: x.set_xlabel("Epoch"); x.grid(alpha=.2, lw=.5)
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, title="Architecture", fontsize=8, title_fontsize=8)
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(os.path.join(a.out, "fig1_training_curves.png"))
