#!/usr/bin/env python3
"""B7 (Fig. 4 and 5) - redraws the figures from xai/xai_per_image.csv, with uncertainty (SD over the 5 seeds). CPU, a few seconds.
Fig. 4: Dice vs references (permuted null, uniform floor; the area-imposed ceiling is printed at the top of each group), deep layer (role=deep), 4 architectures x 3 classes.
Fig. 5: intra-brain AUC per class and seed, 8 configurations (4 archs x deep/head) + mean +- SD over the seeds (right-hand band).
Outputs (in --out): Fig4_dice_references.png/.pdf, Fig5_auc_by_seed.png/.pdf, Fig4_values.csv, Fig5_values.csv
Class palette: blue / orange / aqua (CVD-checked, minimum gap 9.2 dE) + distinct markers (secondary coding).
"""
import argparse, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ARCHS = ["Baseline", "CNN_GAP", "CNN_GAP_BN", "Proposed"]
CLASSES = ["glioma", "meningioma", "pituitary"]
COL = {"glioma": "#2a78d6", "meningioma": "#eb6834", "pituitary": "#1baf7a"}
MRK = {"glioma": "o", "meningioma": "s", "pituitary": "^"}
INK, INK2, GRID = "#1c1c1b", "#5b5a55", "#e6e5e0"
SEEDS = [7, 42, 123, 1337, 2024]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.8})

def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

def seed_means(x, role, col, nan_skip=True):
    g = x[x.role == role].groupby(["arch", "true_class", "seed"])[col]
    return (g.mean() if nan_skip else g.apply(lambda v: v.fillna(0).mean())).rename(col)

def fig4(x, out):
    dice = seed_means(x, "deep", "dice"); null = seed_means(x, "deep", "null_dice")
    floor = seed_means(x, "deep", "uniform_floor"); ceil = seed_means(x, "deep", "ceiling")
    rows = []
    ncls = x[x.role == "deep"].groupby("true_class").stem.nunique()
    top = max(max(dice[a, c].mean() + dice[a, c].std(ddof=1), dice[a, c].max(), null[a, c].mean() + null[a, c].std(ddof=1), null[a, c].max())
              for a in ARCHS for c in CLASSES)
    ylim = float(np.ceil(top * 1.3 / 0.05) * 0.05)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.9), sharey=True)
    w = 0.26
    for ax, arch in zip(axes, ARCHS):
        for i, c in enumerate(CLASSES):
            d, n = dice[arch, c], null[arch, c]; f, ce = floor[arch, c].mean(), ceil[arch, c].mean()
            for j, (vals, kw) in enumerate([(d, dict(color="#2a78d6")), (n, dict(color="#eb6834")),
                                            (pd.Series([f] * len(d)), dict(facecolor="#bdbdb8", edgecolor="white", hatch="///", lw=0))]):
                m, sd = vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0)
                xx = i + (j - 1) * (w + 0.03)
                ax.bar(xx, m, w, **kw, zorder=2)
                if j < 2:
                    ax.errorbar(xx, m, yerr=sd, color=INK, lw=1, capsize=2.5, zorder=4)
                    ax.scatter(np.full(len(vals), xx) + np.linspace(-0.05, 0.05, len(vals)), vals.values, s=7, color="white",
                               edgecolor=INK, lw=0.5, zorder=5)
                rows.append(dict(arch=arch, cls=c, series=["Dice", "permuted null", "uniform floor"][j], mean=m, sd=sd, n_seeds=len(vals)))
            ax.text(i, ylim * 0.985, f"ceiling\n{ce:.2f}", ha="center", va="top", fontsize=7, color=INK2)
            rows.append(dict(arch=arch, cls=c, series="ceiling", mean=ce, sd=ceil[arch, c].std(ddof=1), n_seeds=len(ceil[arch, c])))
            if d.mean() < n.mean():
                ax.text(i, max(d.mean(), n.mean(), f) + 0.035 * ylim / 0.4, "Dice below\nnull", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{c}\n(n={ncls[c]})" for c in CLASSES], fontsize=8); ax.set_title(arch, fontsize=10, color=INK)
        ax.set_ylim(0, ylim); clean(ax)
    axes[0].set_ylabel("Dice (mean over test images of the class)")
    h = [Patch(color="#2a78d6", label="Dice"), Patch(color="#eb6834", label="permuted-mask null"),
         Patch(facecolor="#bdbdb8", edgecolor="white", hatch="///", label="uniform-attention floor"),
         Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=INK, markersize=4, label="single seed")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"): fig.savefig(f"{out}/Fig4_dice_references.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); pd.DataFrame(rows).round(4).to_csv(f"{out}/Fig4_values.csv", index=False)

def fig5(x, out):
    auc = x.groupby(["arch", "role", "true_class", "seed"]).auc_intrabrain.mean()
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4), sharey=True); rows, unstable = [], []
    for r, role in enumerate(["deep", "head"]):
        for k, arch in enumerate(ARCHS):
            ax = axes[r, k]; clean(ax)
            low = [min(CLASSES, key=lambda c: auc[arch, role, c, s]) for s in SEEDS]
            uns = len(set(low)) > 1; unstable.append((arch, role, uns))
            for c in CLASSES:
                v = np.array([auc[arch, role, c, s] for s in SEEDS])
                ax.plot(range(5), v, color=COL[c], lw=1.4, marker=MRK[c], ms=5, zorder=3)
                xm = 5.9 + CLASSES.index(c) * 0.7
                ax.errorbar(xm, v.mean(), yerr=v.std(ddof=1), color=COL[c], marker=MRK[c], ms=6, lw=1.4, capsize=3, zorder=4,
                            markeredgecolor="white", markeredgewidth=0.8)
                rows.append(dict(arch=arch, role=role, cls=c, mean=v.mean(), sd=v.std(ddof=1), **{f"seed{s}": a for s, a in zip(SEEDS, v)}))
            ax.axhline(0.5, color=INK2, lw=0.8, ls=":", zorder=1); ax.axvline(5.1, color=GRID, lw=1)
            ax.set_xticks(list(range(5)) + [5.9, 6.6, 7.3])
            ax.set_xticklabels([str(s) for s in SEEDS] + CLASSES, fontsize=7.5, rotation=45, ha="right")
            ax.set_xlim(-0.4, 7.7); ax.set_ylim(0.3, 0.9)
            ax.set_title(f"{arch} / {role}" + ("  (class order changes)" if uns else ""), fontsize=9, color=INK)
            if uns: ax.title.set_fontweight("bold")
    for r in range(2): axes[r, 0].set_ylabel("intra-brain AUC")
    h = [Line2D([0], [0], color=COL[c], marker=MRK[c], lw=1.4, ms=5, label=c) for c in CLASSES] + \
        [Line2D([0], [0], color=INK2, ls=":", label="chance (0.5)"),
         Line2D([], [], color="none", label="right of each panel: mean ± SD over the 5 seeds")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"): fig.savefig(f"{out}/Fig5_auc_by_seed.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); pd.DataFrame(rows).round(4).to_csv(f"{out}/Fig5_values.csv", index=False)
    print("configurations where the class order changes between seeds:",
          [f"{a}/{r}" for a, r, u in unstable if u], f"({sum(u for _, _, u in unstable)} sur 8)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--xai", required=True); ap.add_argument("--out", default=".")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    x = pd.read_csv(a.xai); x = x[x.true_class.isin(CLASSES)]
    print("rows:", len(x), "| archs:", sorted(x.arch.unique()), "| seeds:", sorted(x.seed.unique()), "| roles:", sorted(x.role.unique()))
    fig4(x, a.out); fig5(x, a.out)
    v4 = pd.read_csv(f"{a.out}/Fig4_values.csv"); print(v4[v4.series == "Dice"].pivot(index="arch", columns="cls", values="mean").round(4))
    print("wrote Fig4/Fig5 (.png, .pdf) and *_values.csv in", a.out)

if __name__ == "__main__": main()
