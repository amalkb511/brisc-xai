#!/usr/bin/env python3
"""B7 (Fig. 2) - Grad-CAM vs expert annotation, 3 cases per class chosen by a RULE FIXED IN ADVANCE (GPU, ~1 min).
Model: Proposed (CNN_GAP_BN_SE), MEDIAN-accuracy run over the 5 seeds (as in the current caption), deep layer 28x28,
         pre-softmax Grad-CAM, float32 (cam_continuous of _xai_common = production path).
Rule (per class, over the 254/306/300 masked test images, from xai/xai_per_image.csv, same run, role=deep):
   score = dice - null_dice  (Dice minus the Dice of a random map of the same area, permuted within the class)
   - "Representative": image whose score is the class MEDIAN (the closest one; tie -> smallest stem)
   - "Best alignment"  : MAXIMUM score
   - "Worst alignment" : MINIMUM score (map furthest below chance)
   Degenerate maps (max <= 1e-12) are excluded from the choice. No case is chosen by hand.
Display: MRI at its original resolution, sharp lesion outline (cyan with a black edge), max-normalised map (0->1) in 'inferno'
            with increasing transparency, common colour scale + bar. Model predictions in the subtitles.
Outputs: Fig2_gradcam_cases_v3.png/.pdf, Fig2_cases_v3.csv   (in --out)
Usage: cd /content/brisc-xai && BRISC_OUT=... python -u B7_fig2.py --xai $BRISC_OUT/xai/xai_per_image.csv --out $BRISC_OUT/B7
"""
import argparse, os
import numpy as np, pandas as pd, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

CLASSES = ["glioma", "meningioma", "pituitary"]
NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]     # manifest y index
CASES = [("representative", "Representative\n(median score)"), ("best", "Best alignment\n(max score)"), ("worst", "Worst alignment\n(min score)")]
INK, INK2 = "#1c1c1b", "#5b5a55"

def select_cases(d):
    """d : xai_per_image rows of the run/role/class. Returns {case: row}."""
    d = d[~d.degenerate.astype(bool)].copy(); d["score"] = d.dice - d.null_dice
    d = d.sort_values(["score", "stem"]).reset_index(drop=True)
    med = d.score.median(); rep = d.iloc[(d.score - med).abs().sort_values(kind="stable").index[0]]
    return {"representative": rep, "best": d.iloc[-1], "worst": d.iloc[0]}

def main():
    import config as C
    from _xai_common import load_inputs, iter_models, roles_for, cam_continuous
    ap = argparse.ArgumentParser(); ap.add_argument("--xai", required=True); ap.add_argument("--out", default=".")
    ap.add_argument("--arch", default="Proposed"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    runs = pd.read_csv(os.path.join(C.TRAIN_DIR, "ablation_runs.csv")); r = runs[runs.arch == a.arch].sort_values("accuracy").reset_index(drop=True)
    seed = int(r.seed.iloc[(len(r) - 1) // 2]); acc = float(r.accuracy.iloc[(len(r) - 1) // 2])
    print(f"{a.arch}: accuracies {r.accuracy.round(4).tolist()} -> run median = seed {seed} (acc {acc:.4f})")
    x = pd.read_csv(a.xai); x = x[(x.arch == a.arch) & (x.seed == seed) & (x.role == "deep")]
    picks, rows = {}, []
    for c in CLASSES:
        sel = select_cases(x[x.true_class == c]); picks[c] = sel
        for k, s in sel.items(): rows.append(dict(cls=c, case=k, stem=s.stem, dice=s.dice, null_dice=s.null_dice, score=s.dice - s.null_dice, lesion_area=s.lesion_area))
    xai, IMGS, MSK, LBL, CIX, BRAIN = load_inputs(); pos = pd.Series(np.arange(len(xai)), index=xai.stem.values)
    order = [(c, k) for c in CLASSES for k, _ in CASES]; idx = np.array([pos[picks[c][k].stem] for c, k in order])
    model = None
    for arch, sd, rid, m in iter_models():
        if arch == a.arch and sd == seed: model = m; break
    assert model is not None, "run not found"
    layer = dict(roles_for(a.arch))["deep"]
    raw, post, up = cam_continuous(model, layer, IMGS[idx], CIX[idx])
    prob = model.predict(IMGS[idx], verbose=0); pred = prob.argmax(1)
    dice_re = np.array([ (2 * np.logical_and(up[i] >= 0.5, MSK[j]).sum() / max((up[i] >= 0.5).sum() + MSK[j].sum(), 1)) for i, j in enumerate(idx)])
    csv_dice = np.array([picks[c][k].dice for c, k in order]); print("max Dice difference, recomputed vs CSV:", float(np.abs(dice_re - csv_dice).max()), "(expected ~0)")
    fig, axes = plt.subplots(3, 6, figsize=(13.2, 7.9), gridspec_kw=dict(wspace=0.04, hspace=0.32))
    outline = [pe.Stroke(linewidth=2.6, foreground="black"), pe.Normal()]
    for n, (c, k) in enumerate(order):
        i, row, col = idx[n], CLASSES.index(c), 2 * [kk for kk, _ in CASES].index(k)
        s = picks[c][k]; img = np.asarray(Image.open(xai.path.iloc[i]).convert("L")); H, W = img.shape
        msk = np.asarray(Image.open(xai.mask_path.iloc[i]).convert("L").resize((W, H), Image.NEAREST)) > 127
        cam = cv2.resize(up[n].astype("float32"), (W, H), interpolation=cv2.INTER_LINEAR).clip(0, 1)
        a0, a1 = axes[row, col], axes[row, col + 1]
        a0.imshow(img, cmap="gray"); a0.contour(msk, levels=[0.5], colors=["#00e5ff"], linewidths=1.2).set_path_effects(outline)
        g = img.astype("float32") / 255.0; al = (0.8 * cam)[..., None]
        a1.imshow((1 - al) * g[..., None] + al * plt.cm.inferno(cam)[..., :3], interpolation="nearest")
        im = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap="inferno")
        a1.contour(msk, levels=[0.5], colors=["#00e5ff"], linewidths=1.2).set_path_effects(outline)
        for ax in (a0, a1): ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
        a1.set_xlabel(f"Dice {s.dice:.2f} · null {s.null_dice:.2f}\npredicted: {NAMES[pred[n]]} ({prob[n, pred[n]]:.2f})", fontsize=7.5, color=INK2, labelpad=3)
        if row == 0:
            a0.set_title(dict(CASES)[k], fontsize=9, color=INK, loc="left", x=0.0)
        if col == 0: a0.set_ylabel(c, fontsize=10, color=INK)
        if row == 0 and col == 0: a0.text(0.02, 0.97, "MRI + lesion", transform=a0.transAxes, color="white", fontsize=7, va="top")
        if row == 0 and col == 0: axes[0, 1].text(0.02, 0.97, "Grad-CAM + lesion", transform=axes[0, 1].transAxes, color="white", fontsize=7, va="top")
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5]); cb = fig.colorbar(im, cax=cax); cb.set_label("Grad-CAM (per-image max = 1)", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"{a.arch}, median-accuracy run (seed {seed}, acc {acc:.3f}), deep layer 28×28, pre-softmax logit of predicted class · cyan outline = expert lesion mask\nCases selected per class on score = Dice − permuted-mask null (median / max / min), non-degenerate maps; not representative of mean performance (see Fig. 4)", fontsize=8.5, color=INK2, y=0.985)
    for ext in ("png", "pdf"): fig.savefig(f"{a.out}/Fig2_gradcam_cases_v3.{ext}", dpi=300, bbox_inches="tight")
    pd.DataFrame(rows).round(4).to_csv(f"{a.out}/Fig2_cases_v3.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False)); print("wrote Fig2_* in", a.out)

if __name__ == "__main__": main()
