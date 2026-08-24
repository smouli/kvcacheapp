"""
Hardware microbenchmarks on Modal GPUs.

Measures HBM-ish memcpy bandwidth, matmul TFLOPS, and (optional) NCCL all-reduce.

  modal run benchmarks/modal_hardware.py
  modal run benchmarks/modal_hardware.py --quick          # T4 only
  modal run benchmarks/modal_hardware.py --with-nccl      # also A100:2 NCCL
"""

from __future__ import annotations

import csv
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "kvcache-benchmarks"
RESULTS_VOL = "kvcache-bench-results"
MOUNT = "/bench-results"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .run_commands(
        "pip install --upgrade pip",
        "pip install torch --index-url https://download.pytorch.org/whl/cu124",
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)


def _device_info(device: int = 0) -> dict:
    import torch

    props = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "memory_gib": round(props.total_memory / (1024**3), 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": int(getattr(props, "multi_processor_count", 0) or 0),
    }


def _memcpy_gbps(nbytes: int = 512 * 1024 * 1024, iters: int = 30) -> float:
    """Device-to-device copy bandwidth (proxy for HBM / copy engine)."""
    import torch

    n = nbytes // 4
    a = torch.empty(n, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return (nbytes * iters / dt) / 1e9


def _matmul_tflops(size: int = 8192, dtype_name: str = "bf16", iters: int = 12) -> float:
    import torch

    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    if dtype_name == "bf16" and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
        dtype_name = "fp16"
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = 2.0 * (size**3)
    return flops / dt / 1e12


def _run_single_gpu(gpu_sku: str) -> dict:
    import torch

    torch.cuda.reset_peak_memory_stats()
    info = _device_info(0)

    # Warmup allocator
    _ = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()

    memcpy = _memcpy_gbps()
    tflops_bf16 = _matmul_tflops(dtype_name="bf16")
    tflops_fp16 = _matmul_tflops(dtype_name="fp16")
    peak = torch.cuda.max_memory_allocated() / (1024**3)

    row = {
        "run_id": f"hw-{uuid.uuid4().hex[:8]}",
        "phase": "H",
        "stack_layer": "hardware",
        "layer": "gpu",
        "provider": "modal",
        "model": info["gpu_name"],
        "engine": f"pytorch+{gpu_sku}",
        "gpu_sku": gpu_sku,
        "gpu_name": info["gpu_name"],
        "memory_gib": info["memory_gib"],
        "compute_capability": info["compute_capability"],
        "sm_count": info["sm_count"],
        "memcpy_gbps": round(memcpy, 1),
        "matmul_tflops_bf16": round(tflops_bf16, 2),
        "matmul_tflops_fp16": round(tflops_fp16, 2),
        "nccl_busbw_gbps": "",
        "nccl_algbw_gbps": "",
        "nccl_world_size": 1,
        "peak_gpu_gib": round(peak, 3),
        "workload": "hardware_probe",
        "concurrency": 1,
        "notes": "Modal hardware microbench: D2D memcpy + GEMM TFLOPS",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    out = Path(MOUNT) / f"{row['run_id']}.json"
    out.write_text(json.dumps(row, indent=2), encoding="utf-8")
    volume.commit()
    return row


def _nccl_worker(rank: int, world_size: int, nbytes: int, result_path: str) -> None:
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    n = nbytes // 4
    t = torch.ones(n, device=f"cuda:{rank}", dtype=torch.float32)
    # Warmup
    for _ in range(5):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iters = 40
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    # Standard bus bandwidth approximation for all-reduce:
    # algbw = size / time; busbw = algbw * 2 * (n-1) / n
    algbw = nbytes / dt / 1e9
    busbw = algbw * 2 * (world_size - 1) / world_size

    if rank == 0:
        Path(result_path).write_text(
            json.dumps(
                {
                    "nccl_algbw_gbps": round(algbw, 2),
                    "nccl_busbw_gbps": round(busbw, 2),
                    "nccl_world_size": world_size,
                    "nccl_iters": iters,
                    "nccl_bytes": nbytes,
                }
            ),
            encoding="utf-8",
        )

    dist.destroy_process_group()


def _run_nccl(gpu_sku: str, world_size: int = 2) -> dict:
    import torch
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() >= world_size, (
        f"need ≥{world_size} GPUs, got {torch.cuda.device_count()}"
    )

    info = _device_info(0)
    result_path = f"/tmp/nccl_result_{uuid.uuid4().hex[:8]}.json"
    nbytes = 256 * 1024 * 1024  # 256 MiB

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        _nccl_worker,
        args=(world_size, nbytes, result_path),
        nprocs=world_size,
        join=True,
    )

    nccl = json.loads(Path(result_path).read_text(encoding="utf-8"))
    row = {
        "run_id": f"hw-{uuid.uuid4().hex[:8]}",
        "phase": "H",
        "stack_layer": "hardware",
        "layer": "nccl",
        "provider": "modal",
        "model": info["gpu_name"],
        "engine": f"pytorch+nccl+{gpu_sku}",
        "gpu_sku": gpu_sku,
        "gpu_name": info["gpu_name"],
        "memory_gib": info["memory_gib"],
        "compute_capability": info["compute_capability"],
        "sm_count": info["sm_count"],
        "memcpy_gbps": "",
        "matmul_tflops_bf16": "",
        "matmul_tflops_fp16": "",
        "nccl_busbw_gbps": nccl["nccl_busbw_gbps"],
        "nccl_algbw_gbps": nccl["nccl_algbw_gbps"],
        "nccl_world_size": nccl["nccl_world_size"],
        "peak_gpu_gib": "",
        "workload": "nccl_allreduce",
        "concurrency": world_size,
        "notes": f"Modal NCCL all-reduce world={world_size}, payload={nbytes}B",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    out = Path(MOUNT) / f"{row['run_id']}.json"
    out.write_text(json.dumps(row, indent=2), encoding="utf-8")
    volume.commit()
    return row


@app.function(gpu="T4", image=image, timeout=600, volumes={MOUNT: volume})
def bench_t4() -> dict:
    return _run_single_gpu("T4")


@app.function(gpu="A10G", image=image, timeout=600, volumes={MOUNT: volume})
def bench_a10g() -> dict:
    return _run_single_gpu("A10G")


@app.function(gpu="A100", image=image, timeout=600, volumes={MOUNT: volume})
def bench_a100() -> dict:
    return _run_single_gpu("A100")


@app.function(gpu="A100:2", image=image, timeout=900, volumes={MOUNT: volume})
def bench_a100_nccl() -> dict:
    return _run_nccl("A100:2", world_size=2)


def _write_csv(rows: list[dict]) -> Path:
    root = Path(__file__).resolve().parents[1]
    out_csv = root / "benchmarks" / "results" / "hardware_runs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    def key(r: dict) -> tuple:
        return (
            r.get("gpu_sku", ""),
            r.get("workload", ""),
            r.get("provider", "modal"),
            str(r.get("nccl_world_size", "")),
        )

    by_key = {key(r): r for r in existing}
    for r in rows:
        by_key[key(r)] = r
    merged = list(by_key.values())

    fieldnames: list[str] = []
    for r in merged:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)

    return out_csv


@app.local_entrypoint()
def main(quick: bool = False, with_nccl: bool = False):
    jobs: list[tuple[str, object]] = [("T4", bench_t4)]
    if not quick:
        jobs.extend(
            [
                ("A10G", bench_a10g),
                ("A100", bench_a100),
            ]
        )
    if with_nccl and not quick:
        jobs.append(("A100:2 NCCL", bench_a100_nccl))

    rows: list[dict] = []
    for label, fn in jobs:
        print(f"→ hardware probe on {label}")
        try:
            row = fn.remote()
            rows.append(row)
            if row.get("workload") == "nccl_allreduce":
                print(
                    f"   NCCL busbw {row.get('nccl_busbw_gbps')} GB/s · "
                    f"algbw {row.get('nccl_algbw_gbps')} GB/s · "
                    f"world={row.get('nccl_world_size')}"
                )
            else:
                print(
                    f"   {row.get('gpu_name')} · "
                    f"memcpy {row.get('memcpy_gbps')} GB/s · "
                    f"bf16 {row.get('matmul_tflops_bf16')} TFLOPS · "
                    f"fp16 {row.get('matmul_tflops_fp16')} TFLOPS"
                )
        except Exception as exc:
            print(f"   FAILED: {exc}")

    if not rows:
        print("No successful hardware runs.")
        return

    out_csv = _write_csv(rows)
    print(f"\nWrote {len(rows)} new rows → {out_csv}")

    import subprocess

    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["python3", str(root / "benchmarks" / "scripts" / "generate_report.py")],
        check=False,
        cwd=str(root),
    )
