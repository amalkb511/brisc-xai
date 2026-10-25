"""
Step 2 — Train the ablation under the frozen protocol.

Four architectures x five seeds (config.PROTOCOL['seeds']), one run per (architecture, seed), on one GPU.
Resumable: a run already recorded in ablation_runs.csv is skipped.

Two protocol points that are not incidental:

  * Brightness augmentation is MULTIPLICATIVE, [0.85, 1.15]. Keras'
    RandomBrightness is ADDITIVE, and on an MRI with a hard zero floor an
    additive offset is clipped asymmetrically: the negative half of the range is
    truncated for background pixels and the positive half is not. The measured
    whole-image shift (+0.0218) matches the analytic prediction
    0.583 x (E[max(0, 0.0042+u)] - 0.0042) = +0.0207 to within 5%. Because a
    global-average-pooling head cannot spatially discount a DC offset while a
    Flatten head can, additive brightness costs the GAP variants 3.6-6.0 points
    of accuracy and the Flatten baseline only 0.9, which is enough to invert the
    ablation ordering. 04_make_tables.py reports this interaction.

  * Epoch cap and early-stopping patience are fixed in config.PROTOCOL (epochs=200, es_patience=20).
    Earlier development runs used a lower cap: At 60 epochs, 6 of 12 runs stopped at the
    cap, meaning early stopping never fired and they were still improving;
    CNN_GAP needed epoch 82 to reach its best validation accuracy.

Checkpoints are staged on LOCAL disk (config.STAGE_DIR) and moved to OUT_ROOT
ONLY after the results row is committed, so an interrupted run cannot leave an
orphan file that blocks the resume. Staging must not be on a network mount:
Drive raises OSError 95 on the write mode Keras uses for .keras files.

Outputs (in TRAIN_DIR): ablation_runs.csv, models/, histories/, predictions/
Run:  python 02_train_ablation.py
"""
import json
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import layers

import config as C
from models import build_cnn, assert_geometry

S = C.PROTOCOL["img_size"]
AUTOTUNE = tf.data.AUTOTUNE

GEOM = tf.keras.Sequential([
    layers.RandomRotation(12 / 360., fill_mode="nearest"),
    layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
    layers.RandomZoom(0.10, 0.10, fill_mode="nearest"),
    layers.RandomFlip("horizontal"),
], name="geometry")


def mult_bright(x):
    f = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.85, 1.15)
    return tf.clip_by_value(x * f, 0.0, 1.0)


def _decode(p, y):
    img = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(p), channels=3), [S, S])
    return tf.cast(img, tf.float32) / 255.0, tf.one_hot(y, len(C.CLASS_NAMES))


def make_ds(df, training, seed=None):
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.y.values.astype("int32")))
    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(C.PROTOCOL["batch_size"])
        ds = ds.map(lambda x, y: (mult_bright(GEOM(x, training=True)), y),
                    num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.batch(C.PROTOCOL["batch_size"])
    return ds.prefetch(AUTOTUNE)


def lr_schedule(epoch, lr):
    w = C.PROTOCOL["warmup_epochs"]
    return C.PROTOCOL["lr"] * (epoch + 1) / w if epoch < w else lr


class StallGuard(tf.keras.callbacks.Callback):
    """Stops a run that is genuinely stuck, not one that is merely slow.

    A fixed val_accuracy threshold at a fixed epoch is calibrated for BatchNorm
    models: it aborted CNN_GAP (which has none) at 0.459 while it was still
    improving monotonically. This fires only on a non-finite loss, or on a run
    that has stopped improving for `patience` epochs AND is still below `floor`.
    """

    def __init__(self, patience=8, floor=0.40, check_from=15):
        super().__init__()
        self.p, self.floor, self.cf = patience, floor, check_from
        self.best, self.stall, self.diverged = -np.inf, 0, False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if not np.isfinite(logs.get("val_loss", 0.0)):
            self.diverged = True
            self.model.stop_training = True
            return
        acc = logs.get("val_accuracy", 0.0)
        if acc > self.best + 1e-4:
            self.best, self.stall = acc, 0
        else:
            self.stall += 1
        if epoch >= self.cf and self.best < self.floor and self.stall >= self.p:
            self.diverged = True
            self.model.stop_training = True


def evaluate(model, ds, df):
    probs = model.predict(ds, verbose=0)
    pred = probs.argmax(1)
    true = df.y.values
    met = dict(accuracy=float((pred == true).mean()),
               macro_f1=float(f1_score(true, pred, average="macro")))
    for i, c in enumerate(C.CLASS_NAMES):
        sel = true == i
        met[f"recall_{c}"] = float((pred[sel] == i).mean()) if sel.any() else np.nan
    return met, true, pred, probs


def train_one(arch, seed, train_df, val_ds, test_ds, test_df):
    run_id = f"{arch}__seed{seed}"
    stage = os.path.join(C.STAGE_DIR, f"{run_id}.keras")
    final = os.path.join(C.TRAIN_DIR, "models", f"{run_id}.keras")

    tf.keras.mixed_precision.set_global_policy(C.TRAIN_POLICY)
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    model = build_cnn(**C.ARCHS[arch], name=arch)
    model.compile(optimizer=tf.keras.optimizers.Adam(C.PROTOCOL["lr"],
                                                     clipnorm=C.PROTOCOL["clipnorm"]),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    guard = StallGuard()
    cbs = [tf.keras.callbacks.EarlyStopping(C.PROTOCOL["monitor"],
                                            patience=C.PROTOCOL["es_patience"],
                                            restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau("val_loss",
                                                factor=C.PROTOCOL["rlr_factor"],
                                                patience=C.PROTOCOL["rlr_patience"],
                                                min_lr=C.PROTOCOL["rlr_min"]),
           tf.keras.callbacks.ModelCheckpoint(stage, monitor=C.PROTOCOL["monitor"],
                                              save_best_only=True),
           tf.keras.callbacks.LearningRateScheduler(lr_schedule),
           tf.keras.callbacks.TerminateOnNaN(), guard]

    print(f"\n{'=' * 72}\n{run_id} | {model.count_params():,} params | "
          f"protocol {C.PROTOCOL_HASH}\n{'=' * 72}")
    t0 = time.time()
    hist = model.fit(make_ds(train_df, True, seed), validation_data=val_ds,
                     epochs=C.PROTOCOL["epochs"], callbacks=cbs, verbose=1)
    minutes = (time.time() - t0) / 60

    if guard.diverged:
        if os.path.exists(stage):
            os.remove(stage)
        del model
        return None

    model.load_weights(stage)
    met, true, pred, probs = evaluate(model, test_ds, test_df)
    va = hist.history["val_accuracy"]
    row = dict(arch=arch, seed=seed, run_id=run_id, params=int(model.count_params()),
               epochs_run=len(va), best_epoch=int(np.argmax(va)) + 1,
               hit_cap=len(va) == C.PROTOCOL["epochs"], minutes=round(minutes, 1),
               diverged=False, **met)
    np.save(os.path.join(C.TRAIN_DIR, "histories", f"{run_id}_pred.npy"), pred)
    json.dump({k: [float(x) for x in v] for k, v in hist.history.items()},
              open(os.path.join(C.TRAIN_DIR, "histories", f"{run_id}.json"), "w"))
    del model
    return row, stage, final


def main():
    C.ensure_dirs()
    assert_geometry()
    json.dump(dict(protocol=C.PROTOCOL, hash=C.PROTOCOL_HASH),
              open(os.path.join(C.TRAIN_DIR, "frozen_protocol.json"), "w"), indent=2)
    print(f"PROTOCOL {C.PROTOCOL['version']} hash={C.PROTOCOL_HASH}")

    mani = pd.read_csv(os.path.join(C.DATASET_DIR, "manifest.csv"))
    train_df = mani[mani.split_i == "train"].reset_index(drop=True)
    val_df = mani[mani.split_i == "val"].reset_index(drop=True)
    test_df = mani[mani.split == "test"].reset_index(drop=True)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    val_ds, test_ds = make_ds(val_df, False), make_ds(test_df, False)

    runs_csv = os.path.join(C.TRAIN_DIR, "ablation_runs.csv")
    rows = pd.read_csv(runs_csv).to_dict("records") if os.path.exists(runs_csv) else []
    done = {(r["arch"], r["seed"]) for r in rows}

    for arch in C.ARCHS:
        for seed in C.PROTOCOL["seeds"]:
            if (arch, seed) in done:
                print(f"skip {arch} seed{seed}")
                continue
            res = train_one(arch, seed, train_df, val_ds, test_ds, test_df)
            if res is None:
                rows.append(dict(arch=arch, seed=seed, run_id=f"{arch}__seed{seed}",
                                 diverged=True))
                C.stamp(pd.DataFrame(rows)).to_csv(runs_csv, index=False)
                continue
            row, stage, final = res
            rows.append(row)
            # commit the row FIRST, then promote the artefact
            C.stamp(pd.DataFrame(rows)).to_csv(runs_csv, index=False)
            # shutil.move, not os.replace: staging is on local disk and the
            # destination is usually a network mount, so a rename cannot work.
            shutil.move(stage, final)
            print(f"  test accuracy = {row['accuracy']:.4f} "
                  f"(best epoch {row['best_epoch']}, cap hit: {row['hit_cap']})")

    df = C.stamp(pd.DataFrame(rows))
    print("\n" + df.groupby("arch").accuracy.agg(["mean", "std", "count"]).round(4).to_string())
    print("\nwrote", runs_csv)


if __name__ == "__main__":
    main()
