"""
Architectures and the corrected Grad-CAM. Shared by steps 2 and 3.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import config as C

S = C.PROTOCOL["img_size"]


def se_block(x, f, ratio=16, name=None):
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(f // ratio, 4), activation="relu")(se)
    se = layers.Dense(f, activation="sigmoid")(se)
    return layers.Multiply(name=name)([x, layers.Reshape((1, 1, f))(se)])


def build_cnn(use_gap=False, use_bn=False, use_se=False, name="m"):
    """Squeeze-and-Excitation is applied BEFORE pooling, so se3 is 56x56 and
    se4 is 28x28. Getting this wrong silently measures the wrong layer, so the
    geometry is asserted in assert_geometry() below."""
    inp = layers.Input((S, S, 3))
    x = inp
    for i, f in enumerate([32, 64, 128, 256]):
        x = layers.Conv2D(f, 3, padding="same", name=f"conv{i+1}")(x)
        if use_bn:
            x = layers.BatchNormalization(name=f"bn{i+1}",
                                          momentum=C.PROTOCOL["bn_momentum"])(x)
        x = layers.Activation("relu", name=f"relu{i+1}")(x)
        if use_se:
            x = se_block(x, f, name=f"se{i+1}")
        x = layers.MaxPooling2D(name=f"pool{i+1}")(x)
    if use_gap:
        x = layers.Conv2D(512, 1, padding="same", name="head_conv")(x)
        if use_bn:
            x = layers.BatchNormalization(name="head_bn",
                                          momentum=C.PROTOCOL["bn_momentum"])(x)
        x = layers.Activation("relu", name="head_relu")(x)
        x = layers.Concatenate(name="gap_gmp_concat")(
            [layers.GlobalAveragePooling2D(name="gap")(x),
             layers.GlobalMaxPooling2D(name="gmp")(x)])
        x = layers.Dense(256, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.4)(x)
    else:
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(256, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(C.CLASS_NAMES), activation="softmax",
                       name="predictions", dtype="float32")(x)
    return models.Model(inp, out, name=name)


def assert_geometry():
    """Fail loudly if a target layer is not the resolution the analysis assumes."""
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("float32")
    m = build_cnn(**C.ARCHS["Proposed"], name="geom")
    seen = {}
    for n, expect in C.EXPECTED_GEOMETRY.items():
        try:
            got = int(m.get_layer(n).output.shape[1])
        except ValueError:
            continue
        seen[n] = got
        assert got == expect, f"layer {n} is {got}x{got}, expected {expect}x{expect}"
    print("layer geometry verified:", seen)
    del m


class CAM:
    """Grad-CAM on the PRE-SOFTMAX class score, in float32.

    Differentiating the post-softmax probability saturates for confidently
    predicted classes and halves the measured alignment for those classes only.
    Two implementation details matter and are easy to get wrong:

      * tape.watch() takes the forward-pass tensor, BEFORE any cast. tf.cast
        creates a new tensor that is off the backward path, so watching the cast
        copy makes tape.gradient return None under mixed_float16.
      * the cast to float32 happens AFTER the backward pass, so that any
        underflow in the fp16 gradient remains measurable rather than hidden.

    raw() returns the pre-ReLU CAM. ReLU, normalisation and thresholding are
    left to the caller so that pre-ReLU diagnostics stay available.
    """

    def __init__(self, model, layer_name):
        self.model = model
        self.pred = model.get_layer("predictions")
        self._act = self.pred.activation
        self.pred.activation = tf.keras.activations.linear
        self.gm = tf.keras.Model(model.inputs,
                                 [model.get_layer(layer_name).output, model.outputs[0]])

    def close(self):
        self.pred.activation = self._act

    def verify(self, x):
        """softmax(logits) must reproduce the model's probabilities."""
        _, out = self.gm(x[:8], training=False)
        got = tf.nn.softmax(tf.cast(out, tf.float32)).numpy()
        self.pred.activation = self._act
        ref = np.asarray(self.model(x[:8], training=False), dtype=np.float32)
        self.pred.activation = tf.keras.activations.linear
        err = float(np.abs(got - ref).max())
        assert err < 2e-2, f"logit head does not reproduce probabilities (max err {err:.2e})"
        return err

    @tf.function(reduce_retracing=True)
    def _cam(self, x, ci):
        with tf.GradientTape() as tape:
            conv, out = self.gm(x, training=False)
            tape.watch(conv)                      # forward tensor, before any cast
            score = tf.gather(tf.cast(out, tf.float32), ci, axis=1, batch_dims=1)
        g = tape.gradient(score, conv)
        if g is None:
            raise RuntimeError("gradient is None: watched tensor is off the backward path")
        g = tf.cast(g, tf.float32)                # cast AFTER the backward pass
        conv = tf.cast(conv, tf.float32)
        return tf.einsum("bhwc,bc->bhw", conv, tf.reduce_mean(g, axis=(1, 2)))

    def raw(self, imgs, cls_idx, batch=16):
        out = []
        for i in range(0, len(imgs), batch):
            out.append(self._cam(tf.convert_to_tensor(imgs[i:i + batch]),
                                 tf.convert_to_tensor(cls_idx[i:i + batch],
                                                      dtype=tf.int32)).numpy())
        return np.concatenate(out)
