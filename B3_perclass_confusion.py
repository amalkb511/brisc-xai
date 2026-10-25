#!/usr/bin/env python3
"""
B3 — Per-class metrics (sensitivity, specificity, precision, recall, F1 + CI) and confusion matrices.

I/O CONTRACT
  INPUT:  --pred CSV: image_id, architecture, seed, y_true, y_pred   (labels = class names)
  OUTPUT: --out CSV: per (architecture, class) -> mean +/- SD over seeds of
           sensitivity (= recall), specificity, precision, F1; + normal 95% CI.
           --conf-dir: one confusion-matrix CSV per architecture, aggregated over seeds.

Purpose: see whether the accuracy loss associated with BN / SE affects one class in particular
(compare the 'sensitivity' columns between CNN_GAP, CNN_GAP_BN, CNN_GAP_BN_SE).
"""
import argparse, os, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix

def per_class_from_cm(cm, labels):
    out={}
    tot=cm.sum()
    for i,l in enumerate(labels):
        tp=cm[i,i]; fn=cm[i,:].sum()-tp; fp=cm[:,i].sum()-tp; tn=tot-tp-fn-fp
        sens=tp/(tp+fn) if tp+fn else np.nan          # = recall
        spec=tn/(tn+fp) if tn+fp else np.nan
        prec=tp/(tp+fp) if tp+fp else np.nan
        f1=2*prec*sens/(prec+sens) if prec and sens and (prec+sens) else np.nan
        out[l]=dict(sensitivity=sens, specificity=spec, precision=prec, f1=f1)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pred",required=True)
    ap.add_argument("--out",default="B3_perclass.csv"); ap.add_argument("--conf-dir",default=".")
    a=ap.parse_args(); df=pd.read_csv(a.pred)
    labels=sorted(pd.unique(df[["y_true","y_pred"]].values.ravel()))
    rows=[]
    for arch,ga in df.groupby("architecture"):
        per_seed={l:{k:[] for k in ["sensitivity","specificity","precision","f1"]} for l in labels}
        cms=[]
        for seed,gs in ga.groupby("seed"):
            cm=confusion_matrix(gs.y_true, gs.y_pred, labels=labels); cms.append(cm)
            pc=per_class_from_cm(cm, labels)
            for l in labels:
                for k in per_seed[l]: per_seed[l][k].append(pc[l][k])
        for l in labels:
            for k,v in per_seed[l].items():
                v=np.array(v,float); m=np.nanmean(v); sd=np.nanstd(v,ddof=1) if len(v)>1 else 0.0
                ci=1.96*sd/np.sqrt(len(v)) if len(v)>1 else 0.0
                rows.append(dict(architecture=arch, **{"class":l}, metric=k,
                    mean=round(m,4), sd=round(sd,4), ci95_lo=round(m-ci,4), ci95_hi=round(m+ci,4)))
        cm_mean=np.mean(cms,axis=0)
        pd.DataFrame(cm_mean, index=labels, columns=labels).to_csv(os.path.join(a.conf_dir,f"B3_confusion_{arch}.csv"))
    pd.DataFrame(rows).to_csv(a.out,index=False); print("wrote",a.out,"+ confusion matrices")

if __name__=="__main__": main()
