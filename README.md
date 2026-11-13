# torch-async-offload

A lightweight PyTorch utility for zero-stall CPU-to-GPU tensor and adapter offloading using asynchronous CUDA streams and pinned host memory.

## Features
- **Double-Buffered VRAM Pipelining:** Overlaps compute on the active CUDA stream with non-blocking DMA host-to-device (`H2D`) transfers on a secondary prefetch stream.
- **Pinned Memory Pool:** Eliminates pageable memory overhead over PCIe on consumer and cloud GPUs (e.g., NVIDIA T4 / L4).
- **Adapter & Expert Swapping:** Designed for dynamic LoRA adapter and sparse MoE layer streaming under tight VRAM constraints.

## Quickstart

```python
import torch
from prefetcher import AsyncVRAMPrefetcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prefetcher = AsyncVRAMPrefetcher(device=device)

# Allocate pinned CPU weights
cpu_weights = [torch.randn(4096, 4096, dtype=torch.float16, pin_memory=True) for _ in range(4)]

# Preload first buffer and pipeline execution
prefetcher.prefetch(cpu_weights[0])
for i in range(len(cpu_weights)):
    active_weight = prefetcher.swap_and_get()
    if i + 1 < len(cpu_weights):
        prefetcher.prefetch(cpu_weights[i + 1])
    # Compute on default stream while next layer transfers in background
    out = active_weight @ active_weight.T
