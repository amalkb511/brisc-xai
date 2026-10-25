#!/usr/bin/env python3
"""
A10 (v2, adapted to this pipeline) — complexity, latency, throughput and GPU memory of the 4 architectures.

Differences from v1: (i) the models are rebuilt with models.build_cnn in float32 and then loaded with load_weights
(exactly as iter_models does for Grad-CAM); (ii) latency is measured on a compiled graph (tf.function), with
synchronisation (.numpy()), not through eager Keras calls (which mostly measure Python overhead); (iii) throughput at
batch 32, the median and the p95, and the environment (GPU, driver, TF) are also recorded.

OUTPUT: --out CSV: arch, params, param_mem_MB (float32 weights = params x 4 bytes), flops_G,
         lat_bs1_ms_{mean,sd,median,p95}, lat_bs32_ms, throughput_bs32_img_s,
         peak_mem_bs1_MB, peak_mem_bs32_MB; plus --env-out (txt): GPU, compute capability, driver, versions.
MEMORY: each architecture is measured in a SEPARATE PROCESS (--only), otherwise the peak depends on the TF allocator
         and on the execution order. Peak = maximum number of bytes allocated by TF on the GPU since the process started
         (weights + activations + cuDNN workspace), read after the bs1 and then the bs32 measurements (cumulative).
FLOPs = floating-point operations of the graph at batch 1 (TF profiler; 1 MAC = 2 FLOPs), input 224x224x3, inference mode.
Run: python -u A10_efficiency.py --out A10_efficiency.csv
      (DO NOT run during training or any other GPU computation: it would distort the latencies.)
"""
import argparse, os, platform, subprocess, time
import numpy as np, pandas as pd, tensorflow as tf
import config as C
from models import build_cnn

S = C.PROTOCOL["img_size"]


def flops_g(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    spec = tf.TensorSpec([1, S, S, 3], tf.float32)
    cf = tf.function(lambda x: model(x, training=False)).get_concrete_function(spec)
    frozen = convert_variables_to_constants_v2(cf)
    with tf.Graph().as_default() as g:
        tf.graph_util.import_graph_def(frozen.graph.as_graph_def(), name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts["output"] = "none"                    # otherwise the profiler prints the whole report
        prof = tf.compat.v1.profiler.profile(g, options=opts)
    return prof.total_float_ops / 1e9


def bench(model, bs, n_warm, n_iter):
    @tf.function
    def fwd(x):
        return model(x, training=False)
    x = tf.random.normal([bs, S, S, 3])
    for _ in range(n_warm):
        fwd(x).numpy()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter(); fwd(x).numpy(); ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts)


def env_info():
    lines = [f"python {platform.python_version()} | tensorflow {tf.__version__}"]
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        d = tf.config.experimental.get_device_details(g)
        lines.append(f"GPU: {d.get('device_name')} | compute capability {d.get('compute_capability')}")
    try:
        lines.append(subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    except Exception:
        pass
    if not gpus:
        lines.append("GPU: none (CPU run)")
    return "\n".join(lines), bool(gpus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="A10_efficiency.csv"); ap.add_argument("--env-out", default="A10_env.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-weights", action="store_true", help="test only (random weights)")
    ap.add_argument("--iters", type=int, default=200); ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--only", default=None, help="a single architecture (used by the multi-process mode)")
    a = ap.parse_args()
    if a.only is None:                      # main mode: one subprocess per architecture
        import sys, tempfile
        parts = []
        for arch in C.ARCHS:
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
            cmd = [sys.executable, "-u", os.path.abspath(__file__), "--only", arch, "--out", tmp,
                   "--env-out", a.env_out, "--seed", str(a.seed), "--iters", str(a.iters), "--warm", str(a.warm)]
            if a.random_weights: cmd.append("--random-weights")
            subprocess.run(cmd, check=True)
            parts.append(pd.read_csv(tmp))
        pd.concat(parts).to_csv(a.out, index=False); print("wrote", a.out, a.env_out, flush=True); return
    env, has_gpu = env_info(); print(env); open(a.env_out, "w").write(env + "\n")
    rows = []
    for arch, spec in [(k, v) for k, v in C.ARCHS.items() if k == a.only]:
        tf.keras.backend.clear_session(); tf.keras.mixed_precision.set_global_policy("float32")
        if has_gpu:
            tf.config.experimental.reset_memory_stats("GPU:0")
        m = build_cnn(**spec, name=arch)
        if not a.random_weights:
            m.load_weights(os.path.join(C.TRAIN_DIR, "models", f"{arch}__seed{a.seed}.keras"))
        params = int(m.count_params())
        try:
            fl = round(flops_g(m), 4)
        except Exception as e:
            fl = float("nan"); print("FLOPs unavailable:", arch, e)
        t1 = bench(m, 1, a.warm, a.iters)
        peak = (tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6) if has_gpu else float("nan")
        t32 = bench(m, 32, 10, max(20, a.iters // 5))
        peak32 = (tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6) if has_gpu else float("nan")
        rows.append(dict(arch=arch, params=params, param_mem_MB=round(params * 4 / 1e6, 2), flops_G=fl,
                         lat_bs1_ms_mean=round(t1.mean(), 3), lat_bs1_ms_sd=round(t1.std(ddof=1), 3),
                         lat_bs1_ms_median=round(float(np.median(t1)), 3), lat_bs1_ms_p95=round(float(np.percentile(t1, 95)), 3),
                         lat_bs32_ms=round(t32.mean(), 3), throughput_bs32_img_s=round(32000 / t32.mean(), 1),
                         peak_mem_bs1_MB=round(peak, 1), peak_mem_bs32_MB=round(peak32, 1)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False); print("wrote", a.out, a.env_out, flush=True)


if __name__ == "__main__":
    main()
