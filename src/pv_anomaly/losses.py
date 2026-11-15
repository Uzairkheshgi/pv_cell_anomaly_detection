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


def soft_skeleton(x: torch.Tensor, iters: int) -> torch.Tensor:
    """Thin a soft mask down to its centre lines, keeping it differentiable.

    Repeatedly erodes the mask with min-pooling and keeps whatever an opening
    would remove at each step, which is the ridge of the shape. This is the
    soft skeleton of Shit et al., "clDice", CVPR 2021 (arXiv 2003.07311).

    Args:
        x: Probabilities or a binary mask, shaped (batch, classes, height, width).
        iters: Erosion steps. Must exceed half the width of the thickest line,
            or the thick parts are erased before their centre is found.

    Returns:
        The soft skeleton, same shape as `x`.
    """

    def erode(t):
        return -F.max_pool2d(-t, 3, 1, 1)

    def dilate(t):
        return F.max_pool2d(t, 3, 1, 1)

    skel = F.relu(x - dilate(erode(x)))
    for _ in range(iters):
        x = erode(x)
        delta = F.relu(x - dilate(erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(
    probs: torch.Tensor, onehot: torch.Tensor, classes: list[int], iters: int
) -> torch.Tensor:
    """Score how well thin structures keep their shape, for the given classes.

    Topology precision is the share of the predicted skeleton that lies inside
    the true mask; topology sensitivity is the share of the true skeleton the
    prediction covers. clDice is their harmonic mean. A crack predicted as
    broken pieces loses sensitivity even when its pixel overlap is high.

    Args:
        probs: Softmax output, shaped (batch, classes, height, width).
        onehot: One-hot target, same shape.
        classes: Class indices to score, e.g. crack and gridline.
        iters: Skeleton erosion steps, see `soft_skeleton`.

    Returns:
        1 − mean clDice over the listed classes present in the batch, or a zero
        that stays on the graph if none of them is present.
    """
    pred, true = probs[:, classes], onehot[:, classes]
    present = true.sum((0, 2, 3)) > 0
    if not present.any():
        return probs.sum() * 0
    pred, true = pred[:, present], true[:, present]
    skel_pred, skel_true = soft_skeleton(pred, iters), soft_skeleton(true, iters)
    dims = (0, 2, 3)
    tprec = ((skel_pred * true).sum(dims) + 1e-6) / (skel_pred.sum(dims) + 1e-6)
    tsens = ((skel_true * pred).sum(dims) + 1e-6) / (skel_true.sum(dims) + 1e-6)
    cl = 2 * tprec * tsens / (tprec + tsens)
    return 1 - cl.mean()


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
        cldice_classes: tuple[int, ...] = (),
        cldice_weight: float = 0.0,
        cldice_iters: int = 5,
        cldice_add: bool = False,
    ):
        """Set up the loss.

        Args:
            num_classes: Number of pixel classes.
            weights: Per-class weights for cross-entropy, or None for equal. (calculated by `median_frequency_weights`).
            alpha: Share of cross-entropy. 0.5 means half each.
            cldice_classes: Thin-structure classes scored by clDice as well as
                Dice. Empty turns clDice off.
            cldice_weight: Share of the overlap term given to clDice; the rest
                stays Dice. 0 turns clDice off.
            cldice_iters: Skeleton erosion steps, see `soft_skeleton`.
            cldice_add: If True, Dice keeps its full weight and clDice is added
                on top, scaled by `cldice_weight`. If False, the two share the
                overlap term.

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
        self.cldice_classes = list(cldice_classes)
        self.cldice_weight, self.cldice_iters = cldice_weight, cldice_iters
        self.cldice_add = cldice_add
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
        overlap = dice_loss
        if self.cldice_weight > 0 and self.cldice_classes:
            # Run the skeleton in float32: under mixed precision the repeated
            # pooling of near-zero probabilities loses too much in float16.
            cl = cldice_loss(
                probs.float(), onehot, self.cldice_classes, self.cldice_iters
            )
            if self.cldice_add:
                # Dice covers every class and clDice only a few, so mixing them
                # took half the Dice signal from all the others (§10.11).
                overlap = dice_loss + self.cldice_weight * cl
            else:
                overlap = (1 - self.cldice_weight) * dice_loss + self.cldice_weight * cl
        return self.alpha * ce + (1 - self.alpha) * overlap
