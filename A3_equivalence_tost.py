#!/usr/bin/env python3
"""
A3 — Equivalence test (TOST) Baseline (Flatten) vs CNN_GAP, margin +/-1 accuracy point.

I/O CONTRACT
  INPUT: a --acc CSV with one row per (architecture, seed):
           columns = architecture, seed, accuracy   (accuracy in POINTS, e.g. 97.66)
           expected architectures: 'Baseline', 'CNN_GAP'  (adjust NAMES if needed)
  OUTPUT: --out CSV with diff, 90% CI, p_TOST (paired + unpaired), superiority, Cohen d,
           and the same analysis EXCLUDING seed 1337 (see A5).

Why paired: the same 5 seeds (42,123,2024,7,1337) are used for both architectures,
so the correct comparison is paired by seed (pair = same seed).
"""
import argparse, math
import numpy as np, pandas as pd
from scipy import stats

NAMES = {"baseline": "Baseline", "gap": "CNN_GAP"}  # adjust if your labels differ
MARGIN = 1.0  # accuracy points

def tost_paired(d, margin):
    """d = vector of paired differences (baseline - gap) per seed."""
    n = len(d); mean = d.mean(); sd = d.std(ddof=1); se = sd/math.sqrt(n); df = n-1
    t_low  = (mean-(-margin))/se; p_low  = 1-stats.t.cdf(t_low, df)
    t_high = (mean-  margin )/se; p_high =   stats.t.cdf(t_high, df)
    tcrit = stats.t.ppf(0.95, df)
    ci = (mean-tcrit*se, mean+tcrit*se)
    t_sup = mean/se; p_sup = 2*(1-stats.t.cdf(abs(t_sup), df))
    return dict(n=n, mean=mean, se=se, df=df, ci90_lo=ci[0], ci90_hi=ci[1],
                p_tost=max(p_low,p_high), equivalent=(max(p_low,p_high)<0.05),
                p_superiority=p_sup, cohen_dz=mean/sd)

def tost_unpaired(a, b, margin):
    mA,sA,nA = a.mean(),a.std(ddof=1),len(a); mB,sB,nB = b.mean(),b.std(ddof=1),len(b)
    diff = mA-mB; se = math.sqrt(sA**2/nA+sB**2/nB)
    df = (sA**2/nA+sB**2/nB)**2/((sA**2/nA)**2/(nA-1)+(sB**2/nB)**2/(nB-1))
    p_low = 1-stats.t.cdf((diff+margin)/se, df); p_high = stats.t.cdf((diff-margin)/se, df)
    tcrit = stats.t.ppf(0.95, df); ci=(diff-tcrit*se, diff+tcrit*se)
    sp = math.sqrt(((nA-1)*sA**2+(nB-1)*sB**2)/(nA+nB-2))
    return dict(n=f"{nA}v{nB}", mean=diff, se=se, df=df, ci90_lo=ci[0], ci90_hi=ci[1],
                p_tost=max(p_low,p_high), equivalent=(max(p_low,p_high)<0.05),
                p_superiority=2*(1-stats.t.cdf(abs(diff/se),df)), cohen_dz=diff/sp)

def run(df, margin, label):
    b = df[df.architecture==NAMES["baseline"]].set_index("seed")["accuracy"]
    g = df[df.architecture==NAMES["gap"]].set_index("seed")["accuracy"]
    common = sorted(set(b.index)&set(g.index))
    d = (b.loc[common]-g.loc[common]).values
    rows=[]
    r=tost_paired(d, margin); r.update(analysis=f"paired_{label}"); rows.append(r)
    r=tost_unpaired(b.loc[common].values, g.loc[common].values, margin); r.update(analysis=f"unpaired_{label}"); rows.append(r)
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--acc",required=True); ap.add_argument("--out",default="A3_tost.csv")
    a=ap.parse_args(); df=pd.read_csv(a.acc)
    out=[]
    out+=run(df, MARGIN, "all_seeds")
    out+=run(df[df.seed!=1337], MARGIN, "no_1337")   # see A5
    res=pd.DataFrame(out)
    for c in ["mean","se","ci90_lo","ci90_hi","cohen_dz"]: res[c]=res[c].round(3)
    for c in ["p_tost","p_superiority"]: res[c]=res[c].round(4)
    res.to_csv(a.out, index=False); print(res.to_string(index=False))
    print(f"\nEquivalence margin = +/-{MARGIN} pt. 'equivalent'=True only if the 90% CI of the difference lies within (-{MARGIN},+{MARGIN}).")

if __name__=="__main__": main()
