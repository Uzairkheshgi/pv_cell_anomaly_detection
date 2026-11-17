"""The training loss: weighted cross-entropy plus Dice.

Defects are about 1% of pixels, so plain cross-entropy can score 99% by
answering "background" everywhere. Weighting makes rare pixels count for more,
and Dice measures overlap while ignoring background entirely.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def median_frequency_weights(
    pixel_counts: np.ndarray, clip: float = 20.0
) -> torch.Tensor:
    """Work out how much each class should count for in the loss.

    A class holding few pixels gets a large weight, so missing it hurts more than
    missing a common one. Classes with no pixels get 0 rather than infinity.

    Args:
        pixel_counts: Pixel count per class.
        clip: Largest allowed weight. Without it, a class holding 0.01% of pixels
            would produce a gradient big enough to destabilise early training.

    Returns:
        One weight per class.

    Raises:
        ValueError: If every count is zero.
    """
    counts = np.asarray(pixel_counts, dtype=float)
    if counts.sum() <= 0:
        raise ValueError("class counts are all zero")
    freq = counts / counts.sum()
    weights = np.zeros_like(freq)
    present = freq > 0
    weights[present] = np.median(freq[present]) / freq[present]
    return torch.from_numpy(np.clip(weights, 0.0, clip)).float()


class CEDiceLoss(nn.Module):
    """Weighted cross-entropy and Dice, mixed by `alpha`.

    Cross-entropy judges each pixel on its own; Dice judges whether the predicted
    shape overlaps the real one. Using both works better than either alone.
    """

    def __init__(
        self, num_classes: int, weights: torch.Tensor | None = None, alpha: float = 0.5
    ):
        """Set up the loss.

        Args:
            num_classes: Number of pixel classes.
            weights: Per-class weights for cross-entropy, or None for equal. (calculated by `median_frequency_weights`).
            alpha: Share of cross-entropy. 0.5 means half each.

        Raises:
            ValueError: If `alpha` is outside 0 to 1, or `weights` is the wrong
                length.
        """
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if weights is not None and weights.numel() != num_classes:
            raise ValueError("weights length must equal num_classes")
        self.num_classes, self.alpha = num_classes, alpha
        self.register_buffer("weights", None if weights is None else weights.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the loss for one batch.

        Args:
            logits: Raw model output, shaped (batch, classes, height, width).
            target: True class per pixel, shaped (batch, height, width).

        Returns:
            A single loss value.
        """
        probs = logits.softmax(1)
        onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        dice = (2 * (probs * onehot).sum(dims) + 1e-6) / (
            probs.sum(dims) + onehot.sum(dims) + 1e-6
        )

        # Classes absent from the batch would otherwise score a free 1.0 and
        # quietly weaken the signal on rare-defect batches.
        present = onehot.sum(dims)[1:] > 0
        dice_loss = 1 - dice[1:][present].mean() if present.any() else logits.sum() * 0
        ce = F.cross_entropy(logits, target, weight=self.weights)
        return self.alpha * ce + (1 - self.alpha) * dice_loss
