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


def inverse_frequency_weights(
    pixel_counts: np.ndarray, clip: float = 20.0
) -> torch.Tensor:
    """Weight each class by the reciprocal of its pixel frequency.

    The most aggressive of the static schemes, and the one FARCLUSS reports as
    causing gradient instability. Included so the benchmark covers the range
    rather than only its middle.

    Args:
        pixel_counts: Pixel count per class.
        clip: Largest allowed weight.

    Returns:
        One weight per class. Absent classes get 0.
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
    """Cui et al.'s class-balanced weights, CVPR 2019 (arXiv 1901.05555).

    Weights by the inverse "effective number" of samples,
    `E_n = (1 - beta^n) / (1 - beta)`, which saturates at `1 / (1 - beta)`.
    **Beta therefore bounds the ratio between the largest and smallest weight**
    -- which is why this method, not any invention of ours, is the published
    answer to median frequency's unbounded floor.

    The caveat that matters here: Cui et al. tuned beta for *sample* counts in
    the hundreds. Ours are *pixel* counts of 1e5 to 1e8, and the useful range
    moves with them. Measured on benchmark_el's 29-class train split, where
    the smallest class holds 78 px:

        beta      0.99   0.999   0.9999   0.99999   0.999999   0.9999999
        spread    1.8:1  13.3:1  129:1    1283:1    12821:1    128195:1

    Median frequency sits at 7735:1, so **beta above about 0.999999 produces a
    wider spread than the scheme it is meant to fix.** On binary PVEL-S, where
    both classes hold ~1e8 pixels, every beta up to 0.9999999 returns uniform
    weights and only 1-1e-9 differentiates. The same beta is therefore useless
    on one dataset and harmful on the other: the standard remedy does not
    transfer to dense prediction without re-tuning by orders of magnitude, and
    that is worth reporting in its own right.

    Args:
        pixel_counts: Pixel count per class.
        beta: In [0, 1). 0 is no re-weighting; closer to 1 approaches inverse
            frequency. The weight ratio is capped at about 1 / (1 - beta).

    Returns:
        One weight per class, normalised to sum to the number of present
        classes. Absent classes get 0.
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
    """Pick a class-weighting scheme by name.

    Args:
        pixel_counts: Pixel count per class, from the training split only.
        scheme: "median", "balanced", "inverse", or "recall".
        beta: Passed to `class_balanced_weights`.
        clip: Passed to the median and inverse schemes.

    Returns:
        One weight per class, or None for "recall", which cannot be computed
        up front because it depends on the model's current predictions.

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

    Cross-entropy judges each pixel on its own; Dice judges whether the predicted
    shape overlaps the real one. Using both works better than either alone.
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
            weights: Per-class weights for cross-entropy, or None for equal. (calculated by `median_frequency_weights`).
            alpha: Share of cross-entropy. 0.5 means half each.
            recall_weighting: Use Tian et al.'s recall loss (arXiv 2106.14917)
                instead of `weights`: each class is weighted by `1 - recall`
                recomputed from the current batch, so a class the model is
                already getting right stops pulling on the gradient. Dynamic,
                so it cannot be precomputed from pixel counts.

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

        # Classes absent from the batch would otherwise score a free 1.0 and
        # quietly weaken the signal on rare-defect batches.
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
                # A class the model already recalls perfectly gets weight 0;
                # one it misses entirely gets 1. Classes absent from the batch
                # get 0 rather than a spurious full weight.
                weights = 1.0 - hit / support.clamp(min=1.0)
                weights[support == 0] = 0.0
        if weights is not None and not weights.any():
            # Every class present in the batch is already recalled perfectly,
            # so every recall weight is 0. Cross-entropy normalises by the sum
            # of the weights, so this would divide by zero and return nan,
            # killing the run. There is genuinely nothing to learn from such a
            # batch, so the cross-entropy term contributes nothing -- written
            # this way rather than as a plain 0.0 to keep the graph connected,
            # the same trick the dice branch above uses.
            ce = logits.sum() * 0
        else:
            ce = F.cross_entropy(logits, target, weight=weights)
        return self.alpha * ce + (1 - self.alpha) * dice_loss
