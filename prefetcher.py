```python
import torch
from typing import Optional


class AsyncVRAMPrefetcher:
    """Double-buffered asynchronous CPU-to-GPU tensor prefetcher using CUDA streams."""

    def __init__(self, device: torch.device):
        self.device = device
        self.transfer_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.active_buffer: Optional[torch.Tensor] = None
        self.shadow_buffer: Optional[torch.Tensor] = None

    def prefetch(self, cpu_tensor: torch.Tensor) -> None:
        """Asynchronously copies a pinned CPU tensor into the shadow VRAM buffer."""
        if self.device.type != "cuda":
            self.shadow_buffer = cpu_tensor
            return

        with torch.cuda.stream(self.transfer_stream):
            self.shadow_buffer = cpu_tensor.to(self.device, non_blocking=True)

    def swap_and_get(self) -> torch.Tensor:
        """Synchronizes the prefetch stream, swaps shadow to active, and returns the tensor."""
        if self.device.type == "cuda" and self.transfer_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)
            if self.shadow_buffer is not None:
                self.shadow_buffer.record_stream(torch.cuda.current_stream(self.device))

        self.active_buffer = self.shadow_buffer
        self.shadow_buffer = None
        if self.active_buffer is None:
            raise RuntimeError("No tensor prefetched into shadow buffer.")
        return self.active_buffer
