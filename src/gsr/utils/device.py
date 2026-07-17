"""Device resolution: 'auto' -> cuda if available else cpu.

Auto-detection lets the exact same run script land on a GPU node (train +
self-warm the cache) or a CPU node (cheap head-training / HP sweeps once
embeddings are cached) with no code change.
"""

from __future__ import annotations


def resolve_device(spec: str) -> str:
    import torch

    have_cuda = torch.cuda.is_available()
    if spec == "auto":
        return "cuda" if have_cuda else "cpu"
    if spec == "cuda" and not have_cuda:
        print("[device] --device cuda requested but no GPU visible; using cpu.")
        return "cpu"
    return spec
