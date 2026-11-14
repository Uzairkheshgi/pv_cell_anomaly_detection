"""The training loss: weighted cross-entropy plus Dice.

Defects are a small share of the pixels, so plain cross-entropy can score well
by answering "background" everywhere. Class weights make rare pixels count for
more, and Dice measures overlap while ignoring background.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def median_frequency_weights(
    pixel_counts: np.ndarray, clip: float = 20.0
) -> torch.Tensor:
    """Weight each class by the median class frequency over its own frequency.

    Rare classes get large weights and common ones small weights. So missing a rare class hurts the loss more than missing a common one.

    Args:
        pixel_counts: Pixel count per class.
        clip: Largest allowed weight.

    Returns:
        One weight per class. Classes with no pixels get 0.

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


def inverse_frequency_weights(
    pixel_counts: np.ndarray, clip: float = 20.0
) -> torch.Tensor:
    """Weight each class by one over its pixel frequency.

    The strongest of the fixed schemes. It is here so the comparison covers the
    extreme case as well.

    Args:
        pixel_counts: Pixel count per class.
        clip: Largest allowed weight.

    Returns:
        One weight per class. Classes with no pixels get 0.
    """
    counts = np.asarray(pixel_counts, dtype=float)
    freq = counts / counts.sum()
    weights = np.zeros_like(freq)
    present = freq > 0
    weights[present] = 1.0 / freq[present]
    weights[present] *= present.sum() / weights[present].sum()
    return torch.from_numpy(np.clip(weights, 0.0, clip)).float()


def class_balanced_weights(
    pixel_counts: np.ndarray, beta: float = 0.9999999
) -> torch.Tensor:
    """Class-balanced weights from Cui et al., CVPR 2019 (arXiv 1901.05555).

    Each class is weighted by one over its "effective number" of pixels,
    `(1 - beta^n) / (1 - beta)`. This number stops growing at `1 / (1 - beta)`,
    so beta limits how far apart the largest and smallest weights can be.

    The right beta depends on the dataset's pixel counts. On benchmark_el,
    0.999 gives a spread of about 13:1 and 0.9999 about 129:1; values above
    roughly 0.999999 spread the weights wider than median frequency does.

    Args:
        pixel_counts: Pixel count per class.
        beta: Between 0 and 1, not including 1. 0 gives equal weights; values
            close to 1 approach inverse frequency.

    Returns:
        One weight per class, scaled so they sum to the number of classes that
        have pixels. Classes with no pixels get 0.
    """
    counts = np.asarray(pixel_counts, dtype=float)
    weights = np.zeros_like(counts)
    present = counts > 0
    effective = (1.0 - np.power(beta, counts[present])) / (1.0 - beta)
    weights[present] = 1.0 / effective
    weights[present] *= present.sum() / weights[present].sum()
    return torch.from_numpy(weights).float()


def class_weights(
    pixel_counts: np.ndarray,
    scheme: str = "median",
    beta: float = 0.9999999,
    clip: float = 20.0,
) -> torch.Tensor | None:
    """Build the class weights for the chosen scheme.

    Args:
        pixel_counts: Pixel count per class, from the training split only.
        scheme: "median", "balanced", "inverse", or "recall".
        beta: Used by "balanced" only.
        clip: Largest allowed weight, used by "median" and "inverse".

    Returns:
        One weight per class, or None for "recall", whose weights depend on the
        model's predictions and are worked out per batch inside the loss.

    Raises:
        ValueError: If `scheme` is not one of the four names.
    """
    if scheme == "median":
        return median_frequency_weights(pixel_counts, clip)
    if scheme == "balanced":
        return class_balanced_weights(pixel_counts, beta)
    if scheme == "inverse":
        return inverse_frequency_weights(pixel_counts, clip)
    if scheme == "recall":
        return None
    raise ValueError(f"unknown weighting scheme: {scheme!r}")


class CEDiceLoss(nn.Module):
    """Weighted cross-entropy and Dice, mixed by `alpha`.

    Cross-entropy scores each pixel on its own. Dice scores how well the
    predicted shape overlaps the real one. The loss uses both.
    """

    def __init__(
        self,
        num_classes: int,
        weights: torch.Tensor | None = None,
        alpha: float = 0.5,
        recall_weighting: bool = False,
    ):
        """Set up the loss.

        Args:
            num_classes: Number of pixel classes.
            weights: Per-class weights for cross-entropy, from `class_weights`.
                None weights every class equally.
            alpha: Share of cross-entropy in the loss. 0.5 means half each.
            recall_weighting: Use recall loss (Tian et al., arXiv 2106.14917)
                instead of `weights`. Each class is weighted by `1 - recall`
                on the current batch, so classes the model already finds
                count for less.

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
        self.recall_weighting = recall_weighting
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

        # Skip classes absent from the batch; they would score a free 1.0.
        present = onehot.sum(dims)[1:] > 0
        dice_loss = 1 - dice[1:][present].mean() if present.any() else logits.sum() * 0
        weights = self.weights
        if self.recall_weighting:
            with torch.no_grad():
                flat, pred = target.reshape(-1), logits.argmax(1).reshape(-1)
                support = torch.bincount(flat, minlength=self.num_classes).float()
                hit = torch.bincount(
                    flat[pred == flat], minlength=self.num_classes
                ).float()
                # Weight is 1 - recall: 0 for a class already found, 1 for one missed.
                # Absent classes get 0.
                weights = 1.0 - hit / support.clamp(min=1.0)
                weights[support == 0] = 0.0
        if weights is not None and not weights.any():
            # All weights are 0 when every class is already found, and cross-entropy
            # would return nan. Use a zero that stays connected to the graph.
            ce = logits.sum() * 0
        else:
            ce = F.cross_entropy(logits, target, weight=weights)
        return self.alpha * ce + (1 - self.alpha) * dice_loss
