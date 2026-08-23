#!/usr/bin/env python3
"""Minimal NCCL all-reduce — run via torchrun --nproc_per_node=N."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    t = torch.tensor([float(rank + 1)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    dist.barrier()

    if rank == 0:
        expected = world * (world + 1) / 2
        ok = abs(t.item() - expected) < 1e-3
        status = "OK" if ok else "FAIL"
        print(f"NCCL all_reduce {status}: got {t.item():.0f}, expected {expected:.0f} ({world} ranks)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
