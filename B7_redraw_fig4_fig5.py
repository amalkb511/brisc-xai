#!/usr/bin/env python3
"""Redraws Figures 4 and 5 of the manuscript from the value tables written by B7_fig45_v2.py
(Fig4_values.csv, Fig5_values.csv). This is the version used for the revised manuscript: no per-seed
points on Fig. 4, and the "order varies" flag on Fig. 5 is computed from the per-seed values.
Usage: python B7_redraw_fig4_fig5.py Fig4_values.csv Fig5_values.csv <out_dir>
"""
import os, sys
V4, V5, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
names={"Baseline":"Baseline_CNN","CNN_GAP":"CNN_GAP","CNN_GAP_BN":"CNN_GAP_BN","Proposed":"CNN_GAP_BN_SE"}
archs=list(names)
# ---- Fig 5
d=pd.read_csv(V5)
S=["seed7","seed42","seed123","seed1337","seed2024"]; lab=["7","42","123","1337","2024"]
col={"glioma":"#2a7ade","meningioma":"#e8643c","pituitary":"#17a673"}; mk={"glioma":"o","meningioma":"s","pituitary":"^"}
fig,axs=plt.subplots(2,4,figsize=(14,6.6),sharey=True)
for r,role in enumerate(["deep","head"]):
    for c,a in enumerate(archs):
        ax=axs[r,c]; g=d[(d.arch==a)&(d.role==role)].set_index("cls")
        ranks={tuple(g[s].sort_values(ascending=False).index) for s in S}
        changes=len(ranks)>1
        for k in col:
            y=[g.loc[k,s] for s in S]
            ax.plot(range(5),y,marker=mk[k],color=col[k],lw=1.6,ms=6)
        ax.axhline(0.5,color="#444",ls=":",lw=1)
        ax.axvline(5.2,color="#ddd",lw=1)
        for i,k in enumerate(col):
            x=6+i*0.9
            ax.errorbar([x],[g.loc[k,"mean"]],yerr=[g.loc[k,"sd"]],fmt=mk[k],color=col[k],ms=6,capsize=3,lw=1.5)
        ax.set_xticks(list(range(5))+[6,6.9,7.8]); ax.set_xticklabels(lab+["glioma","meningioma","pituitary"],rotation=45,ha="right",fontsize=8)
        ax.set_ylim(0.3,0.9); ax.set_xlim(-0.4,8.3)
        ax.set_title(f"{names[a]} / {role}"+("  (order varies)" if changes else ""),fontsize=9.5,fontweight="bold" if changes else "normal")
        ax.grid(alpha=.3,lw=.5); ax.spines[["top","right"]].set_visible(False)
        if c==0: ax.set_ylabel("intra-brain AUC")
from matplotlib.lines import Line2D
h=[Line2D([],[],color=col[k],marker=mk[k],lw=1.6,label=k) for k in col]+[Line2D([],[],color="#444",ls=":",label="chance (0.5)")]
fig.legend(handles=h,loc="lower center",bbox_to_anchor=(0.5,0.045),ncol=4,frameon=False,fontsize=10)
fig.text(0.5,0.012,"Left of each panel: AUC by training seed. Right of each panel: mean ± SD over the 5 seeds. Bold title: the order of the three classes differs between seeds.",ha="center",fontsize=9,color="#444")
fig.tight_layout(rect=[0,0.12,1,1]); fig.savefig(OUT+"/Fig5_auc_by_seed.png",dpi=200); plt.close()
# ---- Fig 4
v=pd.read_csv(V4)
n={"glioma":254,"meningioma":306,"pituitary":300}
fig,axs=plt.subplots(1,4,figsize=(14,4.4),sharey=True)
for c,a in enumerate(archs):
    ax=axs[c]
    for i,k in enumerate(["glioma","meningioma","pituitary"]):
        g=v[(v.arch==a)&(v.cls==k)].set_index("series")
        x=i
        ax.bar(x-0.27,g.loc["Dice","mean"],0.26,color="#2a7ade",yerr=g.loc["Dice","sd"],capsize=3,error_kw={"lw":1.2})
        ax.bar(x,g.loc["permuted null","mean"],0.26,color="#e8643c",yerr=g.loc["permuted null","sd"],capsize=3,error_kw={"lw":1.2})
        ax.bar(x+0.27,g.loc["uniform floor","mean"],0.26,color="#cfcfc8",hatch="///",edgecolor="#9a9a92",lw=0)
        ax.text(x,0.57,f"ceiling\n{g.loc['ceiling','mean']:.2f}",ha="center",va="center",fontsize=8,color="#444")
        if g.loc["Dice","mean"]<min(g.loc["permuted null","mean"],g.loc["uniform floor","mean"]):
            ax.text(x+0.1,0.1,"Dice below\nnull and floor",ha="center",fontsize=7.5,color="#444")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{k}\n(n={n[k]})" for k in n],fontsize=9)
    ax.set_ylim(0,0.65); ax.set_title(names[a],fontsize=11); ax.grid(axis="y",alpha=.3,lw=.5); ax.spines[["top","right"]].set_visible(False)
    if c==0: ax.set_ylabel("Dice (mean over test images of the class)")
h=[plt.Rectangle((0,0),1,1,color="#2a7ade"),plt.Rectangle((0,0),1,1,color="#e8643c"),plt.Rectangle((0,0),1,1,facecolor="#cfcfc8",hatch="///",edgecolor="#9a9a92")]
fig.legend(h,["Dice","permuted-mask null","uniform-attention floor"],loc="lower center",bbox_to_anchor=(0.5,0.045),ncol=3,frameon=False)
fig.text(0.5,0.008,"Bars: mean over 5 seeds; error bars: SD over seeds. Ceiling printed above each group.",ha="center",fontsize=9,color="#444")
fig.tight_layout(rect=[0,0.13,1,1]); fig.savefig(OUT+"/Fig4_dice_references.png",dpi=200); plt.close()
