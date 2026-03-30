"""
Empty-patch skipping wrapper for sliding-window inference on sparse volumes.

When running inference on a stitched multi-tile volume, a significant fraction
of patches fall in empty (black) background regions. This wrapper detects those
patches and returns zeros immediately, avoiding wasteful GPU computation.

Usage:
    model.model = EmptyPatchSkipWrapper(model.model, empty_threshold=0.02)
    # Then proceed with normal SlidingWindowInferer call.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmptyPatchSkipWrapper(nn.Module):
    """
    Wraps a network's forward pass to skip patches that are mostly empty.

    For each patch in the sw_batch:
      - If max(|patch|) < empty_threshold  → return zeros (no GPU kernel)
      - Otherwise                          → run normal forward pass

    Args:
        model:           The wrapped neural network.
        out_channels:    Number of output channels (needed to size the zero tensor).
        empty_threshold: Intensity threshold below which a patch is considered empty.
                         Typical value: 0.01–0.05 (images are normalised to [0,1]).
    """

    def __init__(self, model: nn.Module, out_channels: int, empty_threshold: float = 0.02):
        super().__init__()
        self.model = model
        self.out_channels = out_channels
        self.empty_threshold = empty_threshold

        # Counters (non-tensor, won't interfere with Lightning)
        self._total   = 0
        self._skipped = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, C_in, Z, Y, X)  — MONAI passes sw_batch_size patches at once.
        Returns:  (B, C_out, Z, Y, X)
        """
        B, C_in, Z, Y, X = x.shape
        results = []

        for b in range(B):
            self._total += 1
            patch = x[b]   # (C_in, Z, Y, X)

            if patch.abs().max().item() < self.empty_threshold:
                # Empty patch: return zeros without touching the GPU compute units
                self._skipped += 1
                zeros = torch.zeros(
                    self.out_channels, Z, Y, X,
                    device=x.device, dtype=x.dtype
                )
                results.append(zeros)
            else:
                out = self.model(x[b:b+1])   # (1, C_out, Z, Y, X)
                results.append(out[0])

        return torch.stack(results, dim=0)

    def print_stats(self):
        if self._total > 0:
            skip_pct = self._skipped / self._total * 100
            print(f"  EmptyPatchSkip: {self._skipped}/{self._total} patches skipped "
                  f"({skip_pct:.1f}% saved)")
