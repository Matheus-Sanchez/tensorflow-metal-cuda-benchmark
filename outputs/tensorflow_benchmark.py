#!/usr/bin/env python3
"""Reproducible TensorFlow GPU benchmark for TensorFlow Metal and CUDA.

The script deliberately measures two small, isolated workloads:
  1. FP32 matrix multiplication;
  2. a fixed CNN training step with one synthetic batch already on the GPU.

It never downloads a dataset and does not benchmark tf.data, storage, or host-to-
device copies.  Run ``python tensorflow_benchmark.py --help`` for usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_VERSION = "1.0.0"
BENCHMARK_NAME = "tensorflow-metal-cuda-benchmark"


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M%S%z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert best-effort TensorFlow/system values into JSON-safe data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return repr(value)


def package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def command_output(command: Sequence[str], timeout: int = 10) -> Optional[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def normalize_label(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return result.strip(".-") or "run"


def materialize(tensor: Any) -> Any:
    """Block until a TensorFlow result is available to Python."""
    return tensor.numpy()


def run_many(step, iterations: int) -> Any:
    last = None
    for _ in range(iterations):
        last = step()
    if last is None:
        raise RuntimeError("The benchmark received zero iterations.")
    materialize(last)
    return last


def median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def describe(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty collection.")
    return {
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "max": float(max(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def configured_expectation(args: argparse.Namespace) -> str:
    if args.expect != "auto":
        return args.expect
    return "metal" if platform.system() == "Darwin" else "cuda"


def configure_environment(args: argparse.Namespace) -> None:
    # This must run before TensorFlow import.  set_jit(False) is called again after
    # import because the environment flag alone does not configure every TF build.
    if not args.xla:
        existing = os.environ.get("TF_XLA_FLAGS", "")
        disable_flag = "--tf_xla_auto_jit=0"
        if disable_flag not in existing:
            os.environ["TF_XLA_FLAGS"] = f"{existing} {disable_flag}".strip()


def import_tensorflow():
    try:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed in this Python environment. "
            "Install the TensorFlow configuration already used by the research, "
            "then rerun this script."
        ) from exc
    return tf


def device_details(tf: Any, device: Any) -> Dict[str, Any]:
    try:
        details = tf.config.experimental.get_device_details(device)
    except (AttributeError, RuntimeError, ValueError):
        details = {}
    return json_safe(details)


def collect_runtime_metadata(tf: Any, args: argparse.Namespace) -> Dict[str, Any]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    try:
        build_info = tf.sysconfig.get_build_info()
    except (AttributeError, RuntimeError):
        build_info = {}

    gpu_info = []
    for index, gpu in enumerate(physical_gpus):
        gpu_info.append(
            {
                "index": index,
                "name": gpu.name,
                "device_type": gpu.device_type,
                "details": device_details(tf, gpu),
            }
        )

    return {
        "benchmark": {
            "name": BENCHMARK_NAME,
            "script_version": SCRIPT_VERSION,
            "started_at": utc_now(),
            "expect": configured_expectation(args),
            "xla_enabled": bool(args.xla),
            "verify_placement": bool(args.verify_placement),
            "profile_enabled": bool(args.profile),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            # Hostname is deliberately omitted: it is not necessary for comparison.
        },
        "tensorflow": {
            "version": tf.__version__,
            "keras_version": package_version("keras"),
            "tensorflow_metal_version": package_version("tensorflow-metal"),
            "build_info": json_safe(build_info),
            "physical_gpus": gpu_info,
            "logical_gpus": [device.name for device in logical_gpus],
            "visible_devices": [
                {"name": device.name, "device_type": device.device_type}
                for device in tf.config.get_visible_devices()
            ],
            "xla_jit_setting": json_safe(tf.config.optimizer.get_jit()),
        },
        "environment": {
            "TF_XLA_FLAGS": os.environ.get("TF_XLA_FLAGS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        },
    }


def validate_accelerator(tf: Any, args: argparse.Namespace, metadata: Mapping[str, Any]) -> str:
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        raise RuntimeError(
            "TensorFlow does not expose any GPU. On the Mac, check tensorflow-metal; "
            "on Windows/WSL2, check the CUDA TensorFlow installation and nvidia-smi."
        )

    expectation = configured_expectation(args)
    if expectation == "metal":
        if platform.system() != "Darwin":
            raise RuntimeError("--expect metal requires macOS.")
        metal_version = metadata["tensorflow"]["tensorflow_metal_version"]
        if not metal_version:
            raise RuntimeError(
                "macOS GPU found, but tensorflow-metal is not installed in this environment."
            )
    elif expectation == "cuda":
        if not metadata["environment"]["nvidia_smi"]:
            raise RuntimeError(
                "CUDA GPU was expected but nvidia-smi is unavailable. Run this in the WSL2 "
                "distribution with NVIDIA GPU support, or use --expect any deliberately."
            )
        expected_name = args.require_gpu_name.casefold().strip()
        if expected_name and expected_name not in metadata["environment"]["nvidia_smi"].casefold():
            raise RuntimeError(
                f"The CUDA GPU does not match --require-gpu-name {args.require_gpu_name!r}. "
                f"nvidia-smi reports: {metadata['environment']['nvidia_smi']}"
            )
    return "/GPU:0"


def make_matmul(tf: Any, device: str, matrix_size: int, seed: int, xla: bool):
    with tf.device(device):
        a = tf.random.stateless_normal(
            [matrix_size, matrix_size], seed=[seed, 11], dtype=tf.float32
        )
        b = tf.random.stateless_normal(
            [matrix_size, matrix_size], seed=[seed, 29], dtype=tf.float32
        )
        # Force input allocation to finish before the warm-up/timing phase, without
        # transferring an entire matrix back to the host.
        materialize(tf.reduce_sum(a) + tf.reduce_sum(b))

    @tf.function(jit_compile=xla, reduce_retracing=True)
    def operation(left, right):
        return tf.linalg.matmul(left, right)

    return a, b, operation


def benchmark_matmul(tf: Any, args: argparse.Namespace, device: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    a, b, operation = make_matmul(tf, device, args.matrix_size, args.seed, args.xla)
    for _ in range(args.matmul_warmup):
        materialize(operation(a, b))

    samples: List[Dict[str, Any]] = []
    flops_per_op = 2.0 * float(args.matrix_size) ** 3
    for repetition in range(1, args.repetitions + 1):
        start = time.perf_counter()
        output = run_many(lambda: operation(a, b), args.matmul_iterations)
        elapsed = time.perf_counter() - start
        output_device = output.device
        seconds_per_op = elapsed / args.matmul_iterations
        samples.append(
            {
                "test": "matmul_fp32",
                "repetition": repetition,
                "matrix_size": args.matrix_size,
                "iterations": args.matmul_iterations,
                "elapsed_seconds": elapsed,
                "seconds_per_operation": seconds_per_op,
                "gflops": (flops_per_op / seconds_per_op) / 1e9,
                "output_device": output_device,
            }
        )
    summary = {
        "test": "matmul_fp32",
        "matrix_size": args.matrix_size,
        "iterations": args.matmul_iterations,
        "median_seconds_per_operation": median(
            [sample["seconds_per_operation"] for sample in samples]
        ),
        "median_gflops": median([sample["gflops"] for sample in samples]),
        "elapsed_seconds": describe([sample["elapsed_seconds"] for sample in samples]),
        "output_devices": sorted({sample["output_device"] for sample in samples}),
    }
    return samples, summary


def build_tiny_cnn(tf: Any):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10),
        ],
        name="benchmark_tiny_cnn",
    )


def make_synthetic_batch(tf: Any, device: str, batch_size: int, seed: int):
    with tf.device(device):
        images = tf.random.stateless_normal(
            [batch_size, 28, 28, 1], seed=[seed, batch_size], dtype=tf.float32
        )
        labels = tf.random.stateless_uniform(
            [batch_size],
            seed=[seed + 1, batch_size],
            minval=0,
            maxval=10,
            dtype=tf.int32,
        )
        materialize(tf.reduce_sum(images) + tf.cast(tf.reduce_sum(labels), tf.float32))
    return images, labels


def make_train_step(tf: Any, device: str, images: Any, xla: bool, seed: int):
    # Variables, optimizer slots, and the fixed batch are all created before timing.
    tf.keras.utils.set_random_seed(seed)
    with tf.device(device):
        model = build_tiny_cnn(tf)
        model(images, training=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function(jit_compile=xla, reduce_retracing=True)
    def train_step(features, target):
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            loss = loss_fn(target, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    return model, train_step


def benchmark_training(
    tf: Any, args: argparse.Namespace, device: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Tuple[Any, Any]]:
    samples: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    profile_batch: Optional[Tuple[Any, Any]] = None

    for batch_size in args.batches:
        batch_samples: List[Dict[str, Any]] = []
        for repetition in range(1, args.repetitions + 1):
            images, labels = make_synthetic_batch(tf, device, batch_size, args.seed + repetition)
            _, train_step = make_train_step(
                tf, device, images, args.xla, args.seed + batch_size + repetition
            )
            for _ in range(args.train_warmup):
                run_many(lambda: train_step(images, labels), 1)

            start = time.perf_counter()
            final_loss = run_many(lambda: train_step(images, labels), args.train_steps)
            elapsed = time.perf_counter() - start
            loss_value = float(materialize(final_loss))
            sample = {
                "test": "tiny_cnn_train_resident_batch",
                "repetition": repetition,
                "batch_size": batch_size,
                "iterations": args.train_steps,
                "elapsed_seconds": elapsed,
                "seconds_per_step": elapsed / args.train_steps,
                "steps_per_second": args.train_steps / elapsed,
                "images_per_second": (batch_size * args.train_steps) / elapsed,
                "final_loss": loss_value,
                "input_device": images.device,
                "output_device": final_loss.device,
            }
            samples.append(sample)
            batch_samples.append(sample)
            if batch_size == max(args.batches) and repetition == 1:
                profile_batch = (images, labels)

        summaries.append(
            {
                "test": "tiny_cnn_train_resident_batch",
                "batch_size": batch_size,
                "iterations": args.train_steps,
                "median_seconds_per_step": median(
                    [sample["seconds_per_step"] for sample in batch_samples]
                ),
                "median_steps_per_second": median(
                    [sample["steps_per_second"] for sample in batch_samples]
                ),
                "median_images_per_second": median(
                    [sample["images_per_second"] for sample in batch_samples]
                ),
                "elapsed_seconds": describe(
                    [sample["elapsed_seconds"] for sample in batch_samples]
                ),
                "input_devices": sorted({sample["input_device"] for sample in batch_samples}),
                "output_devices": sorted({sample["output_device"] for sample in batch_samples}),
            }
        )
    if profile_batch is None:
        raise RuntimeError("No training batch was created.")
    return samples, summaries, profile_batch


def profile_training(tf: Any, args: argparse.Namespace, device: str, batch: Tuple[Any, Any], run_dir: Path) -> str:
    images, labels = batch
    _, train_step = make_train_step(tf, device, images, args.xla, args.seed + 99_999)
    profile_dir = run_dir / "profile"
    tf.profiler.experimental.start(str(profile_dir))
    try:
        for _ in range(args.profile_steps):
            run_many(lambda: train_step(images, labels), 1)
    finally:
        tf.profiler.experimental.stop()
    return str(profile_dir)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def run_report(summary: Mapping[str, Any]) -> str:
    metadata = summary["metadata"]
    tf_info = metadata["tensorflow"]
    platform_info = metadata["platform"]
    lines = [
        "# Resultado do benchmark TensorFlow",
        "",
        f"- Executado em: `{metadata['benchmark']['finished_at']}`",
        f"- Sistema: `{platform_info['system']} {platform_info['release']}` ({platform_info['machine']})",
        f"- TensorFlow: `{tf_info['version']}`; tensorflow-metal: `{tf_info['tensorflow_metal_version'] or 'não instalado'}`",
        f"- GPU TensorFlow: `{', '.join(tf_info['logical_gpus']) or 'nenhuma'}`",
        f"- XLA: `{'ativado' if metadata['benchmark']['xla_enabled'] else 'desativado'}`",
        "",
        "## Resultados",
        "",
        "| Teste | Parâmetros | Tempo mediano | Throughput mediano |",
        "|---|---|---:|---:|",
    ]
    for item in summary["result_summaries"]:
        if item["test"] == "matmul_fp32":
            lines.append(
                "| Matmul FP32 | "
                f"{item['matrix_size']}×{item['matrix_size']}, {item['iterations']} operações | "
                f"{format_float(item['median_seconds_per_operation'] * 1000)} ms/op | "
                f"{format_float(item['median_gflops'], 1)} GFLOP/s |"
            )
        else:
            lines.append(
                "| CNN, lote residente | "
                f"batch {item['batch_size']}, {item['iterations']} passos | "
                f"{format_float(item['median_seconds_per_step'] * 1000)} ms/passo | "
                f"{format_float(item['median_images_per_second'], 1)} imagens/s |"
            )
    lines.extend(
        [
            "",
            "## Interpretação",
            "",
            "Este teste exclui download de dataset, DataLoader/tf.data e cópias de dados do "
            "intervalo cronometrado. Portanto, ele isola a execução TensorFlow do backend GPU; "
            "não representa o tempo total de treinamento de um experimento real.",
            "",
            "Consulte `metadata.json` para as versões completas do runtime e `samples.csv` para "
            "todas as repetições individuais.",
        ]
    )
    return "\n".join(lines) + "\n"


def create_run_directory(output_dir: Path, label: str) -> Path:
    run_dir = output_dir / f"tf-benchmark-{iso_timestamp()}-{normalize_label(label)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_benchmark(args: argparse.Namespace) -> int:
    configure_environment(args)
    tf = import_tensorflow()
    tf.config.optimizer.set_jit(bool(args.xla))
    if args.verify_placement:
        tf.debugging.set_log_device_placement(True)

    metadata = collect_runtime_metadata(tf, args)
    device = validate_accelerator(tf, args, metadata)
    run_dir = create_run_directory(Path(args.output_dir), args.label)
    metadata["benchmark"]["selected_device"] = device
    metadata["benchmark"]["run_directory"] = str(run_dir)

    specification = {
        "dtype": "float32",
        "xla_enabled": bool(args.xla),
        "matrix_size": args.matrix_size,
        "matmul_warmup": args.matmul_warmup,
        "matmul_iterations": args.matmul_iterations,
        "train_warmup": args.train_warmup,
        "train_steps": args.train_steps,
        "batches": list(args.batches),
        "repetitions": args.repetitions,
        "model": "Conv2D(16)-MaxPool-Conv2D(32)-GAP-Dense(10)",
        "optimizer": "Adam(learning_rate=0.001)",
        "loss": "SparseCategoricalCrossentropy(from_logits=True)",
        "dataset": "synthetic; fixed batch resident on selected device",
    }

    try:
        matmul_samples, matmul_summary = benchmark_matmul(tf, args, device)
        training_samples, training_summaries, profile_batch = benchmark_training(tf, args, device)
        profile_directory = None
        if args.profile:
            profile_directory = profile_training(tf, args, device, profile_batch, run_dir)
    except Exception:
        # Keep environment information if a backend operation fails, which helps
        # diagnose unsupported Metal operations without presenting a false result.
        metadata["benchmark"]["failed_at"] = utc_now()
        write_json(run_dir / "metadata.json", metadata)
        raise

    metadata["benchmark"]["finished_at"] = utc_now()
    metadata["benchmark"]["profile_directory"] = profile_directory
    all_samples = matmul_samples + training_samples
    summary = {
        "metadata": metadata,
        "benchmark_specification": specification,
        "result_summaries": [matmul_summary] + training_summaries,
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "samples.csv", all_samples)
    (run_dir / "report.md").write_text(run_report(summary), encoding="utf-8")

    print(f"Benchmark concluído: {run_dir}")
    print(f"Resumo: {run_dir / 'report.md'}")
    return 0


def read_summary(location: str) -> Tuple[Path, Dict[str, Any]]:
    path = Path(location)
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.is_file():
        raise RuntimeError(f"summary.json não encontrado em: {path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return summary_path, json.load(handle)


def summary_key(item: Mapping[str, Any]) -> Tuple[Any, ...]:
    if item["test"] == "matmul_fp32":
        return (item["test"], item["matrix_size"], item["iterations"])
    return (item["test"], item["batch_size"], item["iterations"])


def metric_values(item: Mapping[str, Any]) -> Tuple[float, float, str, str]:
    if item["test"] == "matmul_fp32":
        return (
            float(item["median_seconds_per_operation"]),
            float(item["median_gflops"]),
            "s/op",
            "GFLOP/s",
        )
    return (
        float(item["median_seconds_per_step"]),
        float(item["median_images_per_second"]),
        "s/passo",
        "imagens/s",
    )


def comparison_report(comparison: Mapping[str, Any]) -> str:
    mac_meta = comparison["mac"]["metadata"]
    win_meta = comparison["windows"]["metadata"]
    lines = [
        "# Comparativo TensorFlow: Mac Metal vs. Windows/WSL2 CUDA",
        "",
        f"- Mac: TensorFlow `{mac_meta['tensorflow']['version']}`, "
        f"tensorflow-metal `{mac_meta['tensorflow']['tensorflow_metal_version'] or 'não detectado'}`",
        f"- Windows/WSL2: TensorFlow `{win_meta['tensorflow']['version']}`, "
        f"CUDA: `{win_meta['environment']['nvidia_smi'] or 'não detectado'}`",
        "",
        "| Teste | Mac (mediana) | Windows/WSL2 (mediana) | Razão Mac/Windows |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        label = row["test"]
        if row.get("batch_size"):
            label += f" (batch {row['batch_size']})"
        if row.get("matrix_size"):
            label += f" ({row['matrix_size']}×{row['matrix_size']})"
        lines.append(
            f"| {label} | {format_float(row['mac_time'] * 1000)} ms | "
            f"{format_float(row['windows_time'] * 1000)} ms | "
            f"{format_float(row['mac_over_windows_time'], 2)}× |"
        )
    lines.extend(
        [
            "",
            "Uma razão Mac/Windows maior que 1 indica que a execução no Mac demorou mais "
            "neste workload específico. A razão não mede desempenho geral de hardware.",
        ]
    )
    return "\n".join(lines) + "\n"


def compare_runs(args: argparse.Namespace) -> int:
    mac_path, mac_summary = read_summary(args.mac_run)
    win_path, win_summary = read_summary(args.windows_run)
    mac_spec = mac_summary.get("benchmark_specification")
    win_spec = win_summary.get("benchmark_specification")
    if mac_spec != win_spec:
        raise RuntimeError(
            "Os dois resultados usam configurações diferentes. Não é seguro compará-los. "
            f"Mac: {mac_path}; Windows/WSL2: {win_path}"
        )

    mac_results = {summary_key(item): item for item in mac_summary["result_summaries"]}
    win_results = {summary_key(item): item for item in win_summary["result_summaries"]}
    if set(mac_results) != set(win_results):
        raise RuntimeError("Os resultados não contêm exatamente os mesmos testes e parâmetros.")

    rows = []
    for key in sorted(mac_results):
        mac_item = mac_results[key]
        win_item = win_results[key]
        mac_time, mac_throughput, time_unit, throughput_unit = metric_values(mac_item)
        win_time, win_throughput, _, _ = metric_values(win_item)
        row = {
            "test": mac_item["test"],
            "mac_time": mac_time,
            "windows_time": win_time,
            "mac_over_windows_time": mac_time / win_time,
            "mac_throughput": mac_throughput,
            "windows_throughput": win_throughput,
            "windows_over_mac_throughput": win_throughput / mac_throughput,
            "time_unit": time_unit,
            "throughput_unit": throughput_unit,
        }
        if "matrix_size" in mac_item:
            row["matrix_size"] = mac_item["matrix_size"]
        if "batch_size" in mac_item:
            row["batch_size"] = mac_item["batch_size"]
        rows.append(row)

    output_dir = Path(args.output_dir) / f"tf-comparison-{iso_timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    comparison = {
        "created_at": utc_now(),
        "benchmark_specification": mac_spec,
        "mac": {"source": str(mac_path), "metadata": mac_summary["metadata"]},
        "windows": {"source": str(win_path), "metadata": win_summary["metadata"]},
        "rows": rows,
    }
    write_json(output_dir / "comparison.json", comparison)
    write_csv(output_dir / "comparison.csv", rows)
    (output_dir / "comparison.md").write_text(comparison_report(comparison), encoding="utf-8")
    print(f"Comparativo concluído: {output_dir}")
    return 0


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("O valor deve ser inteiro positivo.")
    return result


def add_common_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="benchmark-results", help="Pasta onde o resultado será criado.")
    parser.add_argument("--label", default="run", help="Rótulo curto, por exemplo mac-m4 ou wsl-a2000.")
    parser.add_argument(
        "--expect",
        choices=("auto", "metal", "cuda", "any"),
        default="auto",
        help="Acelerador esperado. auto = Metal no macOS e CUDA nos demais sistemas.",
    )
    parser.add_argument(
        "--require-gpu-name",
        default="",
        help="Opcional: trecho obrigatório no nome retornado por nvidia-smi, por exemplo RTX A2000.",
    )
    parser.add_argument("--matrix-size", type=positive_int, default=4096)
    parser.add_argument("--matmul-warmup", type=positive_int, default=10)
    parser.add_argument("--matmul-iterations", type=positive_int, default=100)
    parser.add_argument("--train-warmup", type=positive_int, default=20)
    parser.add_argument("--train-steps", type=positive_int, default=100)
    parser.add_argument("--batches", type=positive_int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--repetitions", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--xla", action="store_true", help="Ativa XLA explicitamente; o padrão é desativado.")
    parser.add_argument(
        "--verify-placement",
        action="store_true",
        help="Emite no console o posicionamento de operações TensorFlow; não use ao cronometrar resultados finais.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Gera trace TensorBoard do treino em profile/ após as medições principais.",
    )
    parser.add_argument("--profile-steps", type=positive_int, default=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Executa os benchmarks neste computador.")
    add_common_run_options(run_parser)
    compare_parser = subparsers.add_parser("compare", help="Compara um resultado Mac e um Windows/WSL2.")
    compare_parser.add_argument("--mac-run", required=True, help="Diretório da execução Mac ou seu summary.json.")
    compare_parser.add_argument(
        "--windows-run", required=True, help="Diretório da execução Windows/WSL2 ou seu summary.json."
    )
    compare_parser.add_argument("--output-dir", default="benchmark-results", help="Pasta para o comparativo.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return run_benchmark(args)
        return compare_runs(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
