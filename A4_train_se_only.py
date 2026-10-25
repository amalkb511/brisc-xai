#!/usr/bin/env python3
"""
A4 — SE-only ablation (CNN_GAP_SE) under the SAME frozen protocol (hash 6f82d66818c0), 5 seeds.

Reuses 02_train_ablation.py as is (train_one, datasets, callbacks); modifies NEITHER config.py NOR
ablation_runs.csv: everything is written to <BRISC_OUT>/ablation_A4/ (runs csv, models, predictions).
Architecture added AT RUN TIME: GAP + SE, without BatchNorm  (use_gap=True, use_bn=False, use_se=True)
=> completes the 2x2 factorial design: {CNN_GAP, +BN, +SE, +BN+SE (=Proposed)}.
Resumable: a run already recorded in ablation_A4/ablation_runs.csv is skipped.
Run:  python -u A4_train_se_only.py            (from the pipeline folder; ~11-15 min / seed on an L4)
"""
import importlib, os, shutil
import numpy as np, pandas as pd
import config as C

ARCH = "CNN_GAP_SE"
SPEC = dict(use_gap=True, use_bn=False, use_se=True)


def main():
    T = importlib.import_module("02_train_ablation")      # same shared config module (C)
    C.ARCHS[ARCH] = SPEC
    C.TRAIN_DIR = os.path.join(C.OUT_ROOT, "ablation_A4")
    if not hasattr(C, "STAGE_DIR"):
        C.STAGE_DIR = "/content/_stage_A4"                # LOCAL staging (Drive refuses .keras writes)
    os.makedirs(C.STAGE_DIR, exist_ok=True)
    C.ensure_dirs()
    T.assert_geometry()
    print(f"PROTOCOL {C.PROTOCOL['version']} hash={C.PROTOCOL_HASH} | arch={ARCH} {SPEC}", flush=True)

    mani = pd.read_csv(os.path.join(C.DATASET_DIR, "manifest.csv"))
    train_df = mani[mani.split_i == "train"].reset_index(drop=True)
    val_df = mani[mani.split_i == "val"].reset_index(drop=True)
    test_df = mani[mani.split == "test"].reset_index(drop=True)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)
    val_ds, test_ds = T.make_ds(val_df, False), T.make_ds(test_df, False)

    runs_csv = os.path.join(C.TRAIN_DIR, "ablation_runs.csv")
    rows = pd.read_csv(runs_csv).to_dict("records") if os.path.exists(runs_csv) else []
    done = {(r["arch"], r["seed"]) for r in rows}
    for seed in C.PROTOCOL["seeds"]:
        if (ARCH, seed) in done:
            print(f"skip {ARCH} seed{seed}", flush=True); continue
        res = T.train_one(ARCH, seed, train_df, val_ds, test_ds, test_df)
        if res is None:
            rows.append(dict(arch=ARCH, seed=seed, run_id=f"{ARCH}__seed{seed}", diverged=True))
            C.stamp(pd.DataFrame(rows)).to_csv(runs_csv, index=False); continue
        row, stage, final = res
        rows.append(row)
        C.stamp(pd.DataFrame(rows)).to_csv(runs_csv, index=False)   # row first, artifact afterwards
        shutil.move(stage, final)
        print(f"  test accuracy = {row['accuracy']:.4f} (best epoch {row['best_epoch']}, cap hit: {row['hit_cap']})",
              flush=True)
    df = C.stamp(pd.DataFrame(rows))
    print("\n" + df.groupby("arch").accuracy.agg(["mean", "std", "count"]).round(4).to_string())
    print("\nwrote", runs_csv, flush=True)


if __name__ == "__main__":
    main()
