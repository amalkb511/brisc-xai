"""
Step 1 — Build the dataset manifest.

Scans the pristine BRISC download, audits the per-class counts against the
published split, finds byte-identical duplicates by MD5, removes the
cross-partition duplicates from the TRAINING side only, and writes a
deterministic train/val/test manifest.

Two distinct problems are separated here, and only one of them is ours:

  (i)  Local artefacts. A Drive-synced copy of the dataset carried 15 extra
       pituitary training files named "... (1).jpg". They are absent from the
       pristine download; sourcing from DATA_ROOT is the fix, and the count
       assertion below proves it.

  (ii) A property of BRISC 2025 itself. Seven SHA-256 hashes appear in both the
       training and test partitions (five pituitary, two meningioma), affecting
       nine training files because one test image is duplicated three times in
       train. This is verifiable from the authors' own manifest.csv, which ships
       a sha256 column. We remove the affected files from TRAIN and leave TEST
       untouched, so our test metrics remain comparable with work using the
       published 1,000-image benchmark.

Outputs (in DATASET_DIR):
    manifest.csv                 path, stem, label, y, md5, group (perceptual hash), split, split_i, mask_path
    leakage_removed.csv          the nine training files removed, with their test twins
    train_internal_duplicates.csv exact duplicate groups inside train (reported, kept)
    dataset_audit.csv            observed vs published counts

Run:  python 01_build_dataset.py
"""
import hashlib
import os
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C

CLS_ROOT = os.path.join(C.DATA_ROOT, C.CLS_SUBDIR)
SEG_ROOT = os.path.join(C.DATA_ROOT, C.SEG_SUBDIR)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def scan(split):
    rows = []
    for idx, cls in enumerate(C.CLASS_NAMES):
        d = os.path.join(CLS_ROOT, split, cls)
        assert os.path.isdir(d), f"missing class folder {d}"
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append(dict(split=split, path=os.path.join(d, f),
                                 stem=os.path.splitext(f)[0], label=cls, y=idx))
    return pd.DataFrame(rows)


def audit(df, published, name):
    t = pd.DataFrame({"observed": df.groupby("label").size(),
                      "published": pd.Series(published)})
    t["delta"] = t.observed - t.published
    t["split"] = name
    print(f"\n--- {name} ---\n{t[['observed', 'published', 'delta']]}")
    print(f"    total observed={t.observed.sum()} published={t.published.sum()}")
    return t


def phash(path, size=8):
    """Perceptual hash, used only to group near-duplicates so they land on the
    same side of the train/val line. Not used for leakage detection."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size * 4, size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(img))[:size, :size]
    return "".join("1" if v > np.median(dct) else "0" for v in dct.flatten())


def main():
    C.ensure_dirs()
    assert os.path.isdir(CLS_ROOT), (
        f"{CLS_ROOT} not found. Download the pristine dataset first:\n"
        f"  kaggle datasets download -d briscdataset/brisc2025 "
        f"-p {os.path.dirname(C.DATA_ROOT)} --unzip")

    train_pool, test_df = scan("train"), scan("test")

    print("hashing...", end=" ", flush=True)
    t0 = time.time()
    train_pool["md5"] = [md5(p) for p in train_pool.path]
    test_df["md5"] = [md5(p) for p in test_df.path]
    print(f"{len(train_pool) + len(test_df)} files in {time.time() - t0:.0f}s")

    a_tr = audit(train_pool, C.PUBLISHED_BRISC["train"], "train")
    a_te = audit(test_df, C.PUBLISHED_BRISC["test"], "test")
    pd.concat([a_tr, a_te]).to_csv(os.path.join(C.DATASET_DIR, "dataset_audit.csv"))
    ok = bool((a_tr.delta == 0).all() and (a_te.delta == 0).all())
    print("\nMatches the published split:", ok)
    assert ok, ("The copy at DATA_ROOT does not match the published split. It is "
                "not the pristine download - re-download it.")

    # --- cross-partition duplicates: a property of the dataset -------------
    test_md5 = set(test_df.md5)
    leak = train_pool[train_pool.md5.isin(test_md5)].copy()
    leak = leak.merge(
        test_df[["md5", "path", "label"]].rename(
            columns={"path": "test_path", "label": "test_label"}),
        on="md5", how="left")
    print(f"\nTraining files byte-identical to a test image: {len(leak)} "
          f"(distinct test images affected: {leak.md5.nunique()})")
    if len(leak):
        print(leak.groupby("label").size())
    leak.to_csv(os.path.join(C.DATASET_DIR, "leakage_removed.csv"), index=False)

    # --- duplicates inside train: reported, not removed --------------------
    dup = train_pool[train_pool.duplicated("md5", keep=False)]
    print(f"Exact duplicate groups within train: {dup.md5.nunique()} "
          f"covering {len(dup)} files (reported, not removed)")
    dup.to_csv(os.path.join(C.DATASET_DIR, "train_internal_duplicates.csv"), index=False)

    if C.REMOVE_TRAIN_LEAKAGE and len(leak):
        train_pool = train_pool[~train_pool.md5.isin(test_md5)].reset_index(drop=True)
        print(f"\nRemoved {len(leak)} training files. Train-pool counts now:")
        print(train_pool.groupby("label").size())

    # --- masks -------------------------------------------------------------
    lut = {}
    for split in ("train", "test"):
        d = os.path.join(SEG_ROOT, split, "masks")
        if os.path.isdir(d):
            for f in os.listdir(d):
                lut[os.path.splitext(f)[0]] = os.path.join(d, f)
    print(f"\nmasks indexed: {len(lut)}")
    test_df["mask_path"] = test_df.stem.map(lut)
    print("test images with a mask:", int(test_df.mask_path.notna().sum()))
    print(test_df.assign(has=test_df.mask_path.notna())
                 .groupby("label")["has"].agg(["sum", "size"]))

    # --- stratified split, grouped by perceptual hash ----------------------
    if C.USE_PHASH_GROUPING:
        t0 = time.time()
        train_pool["group"] = [phash(p) for p in train_pool.path]
        print(f"phash: {len(train_pool)} -> {train_pool.group.nunique()} groups "
              f"in {time.time() - t0:.0f}s")
    else:
        train_pool["group"] = train_pool.index.astype(str)

    g = train_pool.groupby("group").agg(label=("label", "first")).reset_index()
    _, g_va = train_test_split(g["group"], test_size=C.VAL_FRACTION,
                               stratify=g["label"], random_state=C.SPLIT_SEED)
    train_pool["split_i"] = np.where(train_pool.group.isin(set(g_va)), "val", "train")
    tr = train_pool[train_pool.split_i == "train"]
    va = train_pool[train_pool.split_i == "val"]

    for a, b, na, nb in [(tr, va, "train", "val"), (tr, test_df, "train", "test"),
                         (va, test_df, "val", "test")]:
        assert not (set(a.md5) & set(b.md5)), f"MD5 overlap between {na} and {nb}"
    print(f"\ntrain={len(tr)} val={len(va)} test={len(test_df)} | no MD5 overlap anywhere")

    out = os.path.join(C.DATASET_DIR, "manifest.csv")
    pd.concat([train_pool, test_df], ignore_index=True).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
