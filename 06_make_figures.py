"""
Step 6 — Figures.

Produces the manuscript figures from the CSVs written by steps 1-5, plus one
qualitative panel that needs the frozen models reloaded.

    fig1_qualitative.png   example image / expert mask / Grad-CAM overlay, one row
                           per tumour class, from the median-accuracy run
    fig2_dice_refs.png     Dice against its permuted null and uniform floor, by
                           architecture and class - the three bars where Dice falls
                           below its own null are the point of the figure
    fig3_gradcam_2x2.png   degenerate-map rate by target x precision x class
    fig4_seed_stability.png  per-seed intra-brain AUC, one panel per architecture
                           x layer, showing where the class ordering changes

Figures are written at 300 dpi to XAI_DIR/figures. Fig. 1 needs a GPU and ~2 min;
the rest are CPU-only and take seconds.

Run:  python 06_make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

FIG = os.path.join(C.XAI_DIR, "figures")
DPI = 300
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
ARCH_ORDER = ["Baseline", "CNN_GAP", "CNN_GAP_BN", "Proposed"]
CLS = C.TUMOR_CLASSES


def _order(df, col="arch"):
    present = [a for a in ARCH_ORDER if a in df[col].unique()]
    return present + [a for a in df[col].unique() if a not in present]


# --------------------------------------------------------------------------
def fig2_dice_refs(pi):
    """Dice with the two reference values it needs to be interpretable."""
    d = (pi[pi.role == C.PRIMARY_ROLE]
         .groupby(["arch", "true_class"])[["dice", "null_dice", "uniform_floor"]]
         .mean())
    archs = _order(d.reset_index())
    fig, axes = plt.subplots(1, len(archs), figsize=(9.5, 2.9), sharey=True)
    w = 0.26
    for ax, arch in zip(np.atleast_1d(axes), archs):
        sub = d.loc[arch].reindex(CLS)
        xs = np.arange(len(CLS))
        ax.bar(xs - w, sub.dice, w, label="Dice", color="#2c5f8a")
        ax.bar(xs, sub.null_dice, w, label="permuted null", color="#c0704a")
        ax.bar(xs + w, sub.uniform_floor, w, label="uniform floor",
               color="#9a9a9a", hatch="///", edgecolor="white", linewidth=0)
        for k, cls in enumerate(CLS):
            if sub.dice[cls] < sub.null_dice[cls]:
                ax.annotate("below null", (k, sub.null_dice[cls]),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=6, color="#a02020")
        ax.set_xticks(xs)
        ax.set_xticklabels([c[:4] for c in CLS])
        ax.set_title(arch, fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Dice")
    np.atleast_1d(axes)[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_dice_refs.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  fig2 written")


# --------------------------------------------------------------------------
def fig3_gradcam_2x2(path):
    """Degenerate-map rate under each cell of the target x precision design."""
    g = pd.read_csv(path)
    d = (g.groupby(["true_class", "target", "policy"]).degenerate.mean()
         .rename("rate").reset_index())
    cells = [("logit", "float32"), ("logit", "mixed_float16"),
             ("softmax", "float32"), ("softmax", "mixed_float16")]
    labels = ["logit\nfp32", "logit\nfp16", "softmax\nfp32", "softmax\nfp16"]
    colors = ["#2c5f8a", "#4a86b8", "#c0704a", "#a03020"]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    xs = np.arange(len(CLS))
    w = 0.2
    for i, ((t, p), lab, col) in enumerate(zip(cells, labels, colors)):
        vals = [d[(d.true_class == c) & (d.target == t) & (d.policy == p)]
                .rate.mean() for c in CLS]
        ax.bar(xs + (i - 1.5) * w, vals, w, label=lab, color=col)
    ax.set_xticks(xs)
    ax.set_xticklabels(CLS)
    ax.set_ylabel("fraction of identically zero maps")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_gradcam_2x2.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  fig3 written")


# --------------------------------------------------------------------------
def fig4_seed_stability(pi):
    """Per-seed alignment. Where the lowest line changes class, the ordering is
    not a property of the architecture."""
    d = (pi.groupby(["arch", "role", "seed", "true_class"])[C.PRIMARY_METRIC]
         .mean().reset_index())
    keys = [(a, r) for a in _order(d) for r in ["deep", "head"]
            if not d[(d.arch == a) & (d.role == r)].empty]
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 4.4), sharey=True)
    axes = axes.ravel()
    marks = {"glioma": "o", "meningioma": "s", "pituitary": "^"}
    cols = {"glioma": "#2c5f8a", "meningioma": "#4a8a4a", "pituitary": "#c0704a"}
    for ax, (arch, role) in zip(axes, keys):
        sub = d[(d.arch == arch) & (d.role == role)]
        seeds = sorted(sub.seed.unique())
        lowest = []
        for s in seeds:
            v = sub[sub.seed == s].set_index("true_class")[C.PRIMARY_METRIC]
            lowest.append(v.idxmin())
        for c in CLS:
            v = [sub[(sub.seed == s) & (sub.true_class == c)][C.PRIMARY_METRIC].mean()
                 for s in seeds]
            ax.plot(range(len(seeds)), v, marker=marks[c], color=cols[c],
                    ms=3.5, lw=1, label=c)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        stable = len(set(lowest)) == 1
        ax.set_title(f"{arch} / {role}" + ("" if stable else "  (unstable)"),
                     fontsize=7, color="k" if stable else "#a02020")
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels(seeds, fontsize=6, rotation=45)
    for ax in axes[len(keys):]:
        ax.axis("off")
    axes[0].set_ylabel("intra-brain AUC")
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_seed_stability.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("  fig4 written")


# --------------------------------------------------------------------------
def fig1_qualitative(pi):
    """One row per class: image, expert mask, Grad-CAM overlay. Needs the models."""
    import cv2
    import tensorflow as tf
    from models import CAM, build_cnn
    from importlib.machinery import SourceFileLoader
    m3 = SourceFileLoader("_m3", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "03_measure_xai.py")).load_module()

    S = C.PROTOCOL["img_size"]
    runs = pd.read_csv(os.path.join(C.TRAIN_DIR, "ablation_runs.csv"))
    runs = runs[runs.arch == "Proposed"].sort_values("accuracy")
    rid = runs.iloc[len(runs) // 2]["run_id"]          # median-accuracy run
    w = os.path.join(C.TRAIN_DIR, "models", f"{rid}.keras")

    mani = pd.read_csv(os.path.join(C.DATASET_DIR, "manifest.csv"))
    xai = mani[(mani.split == "test") & mani.mask_path.notna()
               & mani.label.isin(CLS)].reset_index(drop=True)
    # one median-Dice example per class, so the panel is representative
    med = (pi[(pi.role == C.PRIMARY_ROLE) & (pi.run_id == rid)]
           .groupby("true_class").apply(
               lambda g: g.iloc[(g.dice - g.dice.median()).abs().argsort().iloc[0]].stem,
               include_groups=False))
    picks = [xai[xai.stem == med[c]].iloc[0] for c in CLS if c in med.index]

    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.backend.clear_session()
    model = build_cnn(**C.ARCHS["Proposed"], name="Proposed")
    model.load_weights(w)
    cam = CAM(model, C.LAYERS["Proposed"][C.PRIMARY_ROLE])

    fig, axes = plt.subplots(len(picks), 3, figsize=(5.4, 1.85 * len(picks)))
    axes = np.atleast_2d(axes)
    try:
        for r, row in enumerate(picks):
            img = tf.keras.utils.img_to_array(tf.keras.utils.load_img(
                row.path, target_size=(S, S))).astype("float32") / 255.0
            msk = (cv2.resize(cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE),
                              (S, S), interpolation=cv2.INTER_NEAREST) / 255.0) >= 0.5
            raw = cam.raw(img[None], np.array([row.y], "int32"))[0]
            post = np.maximum(raw, 0.0)
            up = tf.image.resize((post / (post.max() + 1e-8))[None, ..., None],
                                 (S, S)).numpy()[0, ..., 0]
            axes[r, 0].imshow(img)
            axes[r, 1].imshow(img[..., 0], cmap="gray")
            axes[r, 1].contour(msk, levels=[0.5], colors="#38d430", linewidths=0.9)
            axes[r, 2].imshow(img[..., 0], cmap="gray")
            axes[r, 2].imshow(up, cmap="jet", alpha=0.45)
            axes[r, 2].contour(msk, levels=[0.5], colors="#38d430", linewidths=0.9)
            axes[r, 0].set_ylabel(row.label, fontsize=8)
            for c in range(3):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    finally:
        cam.close()
    for c, t in enumerate(["MRI slice", "expert mask", "Grad-CAM + mask"]):
        axes[0, c].set_title(t, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_qualitative.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    del model
    print(f"  fig1 written (run {rid})")


def main():
    C.ensure_dirs()
    os.makedirs(FIG, exist_ok=True)
    pi = pd.read_csv(os.path.join(C.XAI_DIR, "xai_per_image.csv"))
    assert set(pi.protocol_hash.dropna().unique()) <= {C.PROTOCOL_HASH}, \
        "xai_per_image.csv mixes protocols"
    print("figures ->", FIG)
    fig2_dice_refs(pi)
    g2x2 = os.path.join(C.XAI_DIR, "gradcam_2x2.csv")
    if os.path.exists(g2x2):
        fig3_gradcam_2x2(g2x2)
    else:
        print("  fig3 skipped: run 05_gradcam_target_2x2.py first")
    fig4_seed_stability(pi)
    try:
        fig1_qualitative(pi)
    except Exception as e:      # the qualitative panel is the only GPU-dependent one
        print(f"  fig1 failed ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
