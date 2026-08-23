#!/usr/bin/env python3
"""Print CUDA / NCCL build info from PyTorch (single-node check)."""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() == False")
        return 1

    n = torch.cuda.device_count()
    print(f"CUDA available: True")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
    nccl = getattr(torch.cuda, "nccl", None)
    if nccl and hasattr(nccl, "version"):
        try:
            print(f"NCCL version: {nccl.version()}")
        except Exception as exc:
            print(f"NCCL version: (query failed: {exc})")
    else:
        print("NCCL: bundled with torch.cuda (no version API)")

    print(f"GPU count: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        name = torch.cuda.get_device_name(i)
        mem_gib = props.total_memory / (1024**3)
        print(f"  [{i}] {name} · {mem_gib:.1f} GiB · CC {props.major}.{props.minor}")

    # Quick single-GPU matmul
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"Single-GPU bf16 matmul OK ({y.shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
