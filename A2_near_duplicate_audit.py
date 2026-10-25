#!/usr/bin/env python3
"""
A2 — Near-duplicates across partitions (pHash) + impact on accuracy.

I/O CONTRACT
  INPUT:
    --train-dir, --test-dir: image folders (subfolders = classes) OR
    --phash-train, --phash-test: precomputed CSVs (columns: path, class, phash_hex)
    --pred: CSV of test predictions (columns: path, class, correct[0/1], model, seed)
  OUTPUT: --out CSV -> for each Hamming threshold d in THRESHOLDS:
    number of test images at distance <= d from a train image, per class;
    accuracy on the 'near' vs 'far' subset, per (model, seed) and per class.

Note: 64-bit pHash (8x8 DCT), identical to the description in Section II-A/2 of the manuscript.
"""
import argparse, os, itertools
import numpy as np, pandas as pd
THRESHOLDS=[0,2,4,6,8,10]

def phash_dir(d):
    import imagehash; from PIL import Image
    rows=[]
    for cls in sorted(os.listdir(d)):
        cd=os.path.join(d,cls)
        if not os.path.isdir(cd): continue
        for f in os.listdir(cd):
            p=os.path.join(cd,f)
            try: h=imagehash.phash(Image.open(p).convert("L"), hash_size=8)
            except Exception: continue
            rows.append(dict(path=p, **{"class":cls}, phash_hex=str(h)))
    return pd.DataFrame(rows)

def to_bits(hx):  # hex -> 64-bit 0/1 vector
    return np.array([int(b) for b in bin(int(hx,16))[2:].zfill(64)], dtype=np.uint8)

def min_hamming(test_bits, train_bits):
    # minimum Hamming distance from each test image to the train set (vectorised)
    out=np.empty(len(test_bits), dtype=np.int16)
    for i,tb in enumerate(test_bits):
        out[i]=(train_bits ^ tb).sum(axis=1).min()
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-dir"); ap.add_argument("--test-dir")
    ap.add_argument("--phash-train"); ap.add_argument("--phash-test")
    ap.add_argument("--pred", required=True); ap.add_argument("--out", default="A2_neardup.csv")
    a=ap.parse_args()
    tr = pd.read_csv(a.phash_train) if a.phash_train else phash_dir(a.train_dir)
    te = pd.read_csv(a.phash_test)  if a.phash_test  else phash_dir(a.test_dir)
    trb=np.stack(tr.phash_hex.map(to_bits)); teb=np.stack(te.phash_hex.map(to_bits))
    te=te.copy(); te["dmin"]=min_hamming(list(teb), trb)
    pred=pd.read_csv(a.pred)
    key=[c for c in ["path"] if c in pred.columns] or ["path"]
    pred=pred.merge(te[["path","dmin","class"]].rename(columns={"class":"class_te"}), on="path", how="left")
    rows=[]
    for d in THRESHOLDS:
        near = pred.dmin<=d
        for (model,seed),g in pred.groupby(["model","seed"]):
            for cls in ["all"]+sorted(g.class_te.dropna().unique()):
                gg = g if cls=="all" else g[g.class_te==cls]
                gn, gf = gg[gg.dmin<=d], gg[gg.dmin>d]
                rows.append(dict(threshold=d, model=model, seed=seed, cls=cls,
                    n_near=len(gn), n_far=len(gf),
                    acc_near=round(gn.correct.mean(),4) if len(gn) else np.nan,
                    acc_far =round(gf.correct.mean(),4) if len(gf) else np.nan,
                    delta=round((gn.correct.mean()-gf.correct.mean()),4) if len(gn) and len(gf) else np.nan))
    # global summary of the counts (model-independent)
    for d in THRESHOLDS:
        sub=te[te.dmin<=d]
        rows.append(dict(threshold=d, model="_count", seed=-1, cls="all",
            n_near=len(sub), n_far=len(te)-len(sub), acc_near=np.nan, acc_far=np.nan, delta=np.nan))
    pd.DataFrame(rows).to_csv(a.out,index=False); print("wrote", a.out)

if __name__=="__main__": main()
