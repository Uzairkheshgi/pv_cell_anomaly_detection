"""Scoring, reported per class.

Overall pixel accuracy is left out on purpose. Background is about 97% of pixels,
so a model that finds nothing still scores 97% and looks fine.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SIZE_BINS = (100, 600, 2_000, 10_000)


SIZE_EDGES = np.array([0, *SIZE_BINS, np.inf])


def _add_regions(target, pred, hits, totals) -> None:
    """Add one image's defect regions to the running size tally.

    Args:
        target: True mask.
        pred: Predicted mask.
        hits: Per-bucket count of regions found. Updated in place.
        totals: Per-bucket count of regions present. Updated in place.
    """
    n_labels, labels = cv2.connectedComponents(
        (target > 0).astype(np.uint8), connectivity=8
    )
    for label in range(1, n_labels):
        region = labels == label
        bucket = int(np.searchsorted(SIZE_EDGES, region.sum(), "right") - 1)
        totals[bucket] += 1
        # Counted as detected only if covered, not merely touched.
        hits[bucket] += (pred[region] > 0).mean() >= 0.5


def _buckets(hits, totals) -> list[dict]:
    """Turn the running tallies into a list of per-size-bucket results."""
    return [
        {
            "min": int(SIZE_EDGES[i]),
            "max": float(SIZE_EDGES[i + 1]),
            "n": int(totals[i]),
            "recall": float(hits[i] / totals[i]) if totals[i] else float("nan"),
        }
        for i in range(len(totals))
    ]


def size_recall(targets, preds) -> list[dict]:
    """Measure what share of defects was found, grouped by defect size.

    This is the diagnostic that says whether the tile size is earning its keep.
    Small defects are the hard ones, and one overall score hides that.

    Args:
        targets: True masks, one per image.
        preds: Predicted masks, one per image.

    Returns:
        One entry per size bucket, with its pixel range, how many defects fell in
        it, and the share of those that were found.
    """
    hits, totals = (
        np.zeros(len(SIZE_EDGES) - 1, int),
        np.zeros(len(SIZE_EDGES) - 1, int),
    )
    for target, pred in zip(targets, preds):
        _add_regions(target, pred, hits, totals)
    return _buckets(hits, totals)


def evaluate(
    stream, num_classes: int, score_pixels: int = 20_000, seed: int = 0
) -> dict:
    """Score predictions, reading one image at a time.

    Images are never kept: the full test set is about 2 GB of arrays, and holding
    it all to score at the end costs several GB more.

    Args:
        stream: Yields (target, pred, score) per image, where `score` is the
            predicted chance of defect per pixel.
        num_classes: Number of pixel classes.
        score_pixels: Pixels sampled per image for AUPRC and AUROC. Sampling is
            uniform, so the defect rate is preserved and the estimate matches
            scoring every pixel to several decimal places.
        seed: Random seed for that sampling.

    Returns:
        A dict of per-class IoU and F1, their averages, AUPRC, AUROC, and the
        size breakdown. AUPRC and AUROC are None if the sample held only one
        class, since both are undefined then.
    """
    n = num_classes
    cm = np.zeros((n, n), np.int64)
    hits, totals = (
        np.zeros(len(SIZE_EDGES) - 1, int),
        np.zeros(len(SIZE_EDGES) - 1, int),
    )
    rng = np.random.default_rng(seed)
    y_true, y_score = [], []

    for target, pred, score in stream:
        cm += np.bincount(target.ravel() * n + pred.ravel(), minlength=n * n).reshape(
            n, n
        )
        _add_regions(target, pred, hits, totals)
        if score_pixels:
            flat = target.size
            idx = rng.integers(0, flat, min(score_pixels, flat))
            y_true.append((target.ravel() > 0)[idx])
            y_score.append(score.ravel()[idx])

    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(invalid="ignore"):
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)

    out = {
        "iou": iou.tolist(),
        "f1": f1.tolist(),
        "support": cm.sum(1).tolist(),
        "mean_iou": float(np.nanmean(iou)),
        "macro_f1": float(np.nanmean(f1)),
        "auprc": None,
        "auroc": None,
        "size_recall": _buckets(hits, totals),
    }

    # AUPRC is the right summary under this imbalance, but both it and AUROC
    # are undefined when the sample contains only one class.
    if y_true:
        y_true, y_score = np.concatenate(y_true), np.concatenate(y_score)
        if y_true.min() != y_true.max():
            out["auprc"] = float(average_precision_score(y_true, y_score))
            out["auroc"] = float(roc_auc_score(y_true, y_score))
    return out


def report(metrics: dict, class_names: list[str]) -> str:
    """Format the scores as a table for printing.

    Args:
        metrics: The dict from `evaluate`.
        class_names: One name per class.

    Returns:
        A multi-line string.
    """
    lines = [f"{'class':<12}{'IoU':>8}{'F1':>8}{'support':>14}"]
    for name, iou, f1, support in zip(
        class_names, metrics["iou"], metrics["f1"], metrics["support"]
    ):
        lines.append(f"{name:<12}{iou * 100:>8.2f}{f1 * 100:>8.2f}{support:>14,}")
    lines.append(f"{'mean IoU':<12}{metrics['mean_iou'] * 100:>8.2f}")
    lines.append(f"{'macro F1':<12}{metrics['macro_f1'] * 100:>8.2f}")
    for key in ("auprc", "auroc"):
        if metrics[key] is not None:
            lines.append(f"{'pixel ' + key.upper():<12}{metrics[key] * 100:>8.2f}")
    for bucket in metrics["size_recall"]:
        if bucket["n"]:
            high = "inf" if np.isinf(bucket["max"]) else f"{int(bucket['max'])}"
            lines.append(
                f"  {bucket['min']:>6}-{high:<6} n={bucket['n']:<5} "
                f"recall={bucket['recall'] * 100:5.1f}%"
            )
    return "\n".join(lines)
