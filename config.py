"""
Central configuration. EDIT THIS FILE ONLY.

Every path below is a directory on your machine or Drive. Nothing else in the
repository contains a hard-coded path.

The PROTOCOL dict is hashed and the hash is stamped into every output row.
Changing any value here changes the hash, which makes runs from different
protocols impossible to pool by accident. Do not edit PROTOCOL to reproduce
published numbers.
"""
import hashlib
import json
import os
import tempfile

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Pristine BRISC 2025 download. Get it with:
#   kaggle datasets download -d briscdataset/brisc2025 -p <parent> --unzip
# Do NOT point this at a copy that has been through a Drive/Dropbox sync: that
# is how the 15 spurious "... (1).jpg" duplicates appeared in our first run.
DATA_ROOT = os.environ.get("BRISC_ROOT", "/content/brisc_fresh/brisc2025")

# Everything this repository writes goes under here.
OUT_ROOT = os.environ.get("BRISC_OUT", "./results")

CLS_SUBDIR = "classification_task"
SEG_SUBDIR = "segmentation_task"

DATASET_DIR = os.path.join(OUT_ROOT, "dataset")
TRAIN_DIR = os.path.join(OUT_ROOT, "ablation")
XAI_DIR = os.path.join(OUT_ROOT, "xai")
TABLES_DIR = os.path.join(OUT_ROOT, "tables")

# Checkpoints are staged on LOCAL disk, never on a network mount. Google Drive
# does not support the write mode Keras uses to save a .keras file and raises
# OSError 95 (Operation not supported). Only the finished artefact is moved to
# OUT_ROOT, which also avoids writing a 158 MB checkpoint to Drive at every
# improving epoch.
STAGE_DIR = os.environ.get("BRISC_STAGE", os.path.join(tempfile.gettempdir(), "brisc_stage"))

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
TUMOR_CLASSES = ["glioma", "meningioma", "pituitary"]

# Fateh et al., Scientific Data 2026, DOI 10.1038/s41597-026-06753-y, CC BY 4.0
PUBLISHED_BRISC = {
    "train": {"glioma": 1147, "meningioma": 1329, "no_tumor": 1067, "pituitary": 1457},
    "test": {"glioma": 254, "meningioma": 306, "no_tumor": 140, "pituitary": 300},
}
DATASET_DOI = "10.1038/s41597-026-06753-y"
DATASET_LICENSE = "CC BY 4.0"

VAL_FRACTION = 0.15
SPLIT_SEED = 20260902
REMOVE_TRAIN_LEAKAGE = True   # train side only; the test set stays the published benchmark
USE_PHASH_GROUPING = True     # keeps near-duplicate images on one side of the train/val line

# --------------------------------------------------------------------------
# Frozen training protocol
# --------------------------------------------------------------------------
PROTOCOL = dict(
    version="frozen-2026-09-13-5seeds",
    augmentation="geometric + MULTIPLICATIVE brightness [0.85, 1.15]",
    img_size=224,
    batch_size=32,
    epochs=200,
    es_patience=20,
    rlr_factor=0.3,
    rlr_patience=8,
    rlr_min=1e-7,
    lr=5e-4,
    clipnorm=1.0,
    warmup_epochs=3,
    bn_momentum=0.9,
    monitor="val_accuracy",
    seeds=[42, 123, 2024, 7, 1337],
)
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()[:12]

TRAIN_POLICY = "mixed_float16"   # Grad-CAM is always computed in float32; see 03_measure_xai.py

ARCHS = {
    "Baseline":   dict(use_gap=False, use_bn=False, use_se=False),
    "CNN_GAP":    dict(use_gap=True,  use_bn=False, use_se=False),
    "CNN_GAP_BN": dict(use_gap=True,  use_bn=True,  use_se=False),
    "Proposed":   dict(use_gap=True,  use_bn=True,  use_se=True),
}

# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------
# Layer ROLES, not names, so the two-depth analysis exists in every architecture.
#   deep = last 28x28 feature map        head = last 14x14 map before the classifier
# SE precedes pooling, so se4 is 28x28 and se3 is 56x56. Asserted at build time.
LAYERS = {
    "Baseline":   {"deep": "relu4", "head": "pool4"},
    "CNN_GAP":    {"deep": "relu4", "head": "head_relu"},
    "CNN_GAP_BN": {"deep": "relu4", "head": "head_relu"},
    "Proposed":   {"deep": "se4",   "head": "head_relu"},
}
EXPECTED_GEOMETRY = {"se3": 56, "se4": 28, "relu4": 28, "head_relu": 14, "pool4": 14}

THRESHOLD = 0.5
NULL_REPS = 5        # within-class mask permutations
N_BOOT = 5000        # bootstrap resamples for the CI on a paired difference
ANALYSIS_SEED = 20260911
BRAIN_MIN_FRAC = 0.15

PRIMARY_ROLE = "deep"       # pre-specified: the finest layer at which every class has >= 5
PRIMARY_METRIC = "auc_intrabrain"   # expected positive cells; head is a sensitivity analysis


def ensure_dirs():
    for d in (OUT_ROOT, DATASET_DIR, TRAIN_DIR, XAI_DIR, TABLES_DIR,
              os.path.join(TRAIN_DIR, "models"), STAGE_DIR,
              os.path.join(TRAIN_DIR, "histories")):
        os.makedirs(d, exist_ok=True)


def stamp(df):
    """Every results table carries the protocol hash. Rows with different hashes
    come from different protocols and must never be pooled."""
    df = df.copy()
    df["protocol_hash"] = PROTOCOL_HASH
    return df
