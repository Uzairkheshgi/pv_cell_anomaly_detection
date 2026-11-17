"""Run a trained model on photos and save pictures of what it marks as defect."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from pv_anomaly.data import augmentations, crop, stitch, tile_positions
from pv_anomaly.train import resolve_device


def load_model(checkpoint: Path, device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model from a best.pt checkpoint.

    Args:
        checkpoint: A best.pt written by training; it carries the run config.
        device: Where to run.

    Returns:
        The model in eval mode, and the run config as a dict.
    """
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = state["config"]
    model = smp.create_model(
        cfg["arch"],
        encoder_name=cfg["encoder"],
        encoder_weights=None,
        in_channels=3,
        classes=cfg["num_classes"],
    )
    # Weight decay leaves ~24M weights as denormal floats. Harmless on a GPU,
    # but on a CPU they make one tile take ~40s instead of ~0.3s, so zero them.
    weights = state["model"]
    for value in weights.values():
        if value.is_floating_point():
            value[value.abs() < torch.finfo(value.dtype).tiny] = 0
    model.load_state_dict(weights)
    return model.to(device).eval(), cfg


@torch.no_grad()
def segment(model, image: np.ndarray, cfg: dict, device) -> np.ndarray:
    """Predict a defect probability for every pixel of one photo.

    Same tiling and stitching as training's `predict`, without needing a mask.

    Args:
        model: The trained model.
        image: RGB uint8 array shaped (height, width, 3).
        cfg: Run config from the checkpoint.
        device: Where to run.

    Returns:
        Defect probability per pixel, shaped (height, width).
    """
    transform = augmentations(train=False)
    height, width = image.shape[:2]
    positions = tile_positions(height, width, cfg["tile"], cfg["stride"])
    logits = []
    for start in range(0, len(positions), cfg["batch_size"]):
        chunk = positions[start : start + cfg["batch_size"]]
        tiles = [
            transform(image=crop(image, y, x, cfg["tile"]))["image"] for y, x in chunk
        ]
        batch = torch.from_numpy(np.stack(tiles).transpose(0, 3, 1, 2)).to(device)
        logits.extend(model(batch.float()).float().cpu().numpy())
    probs = torch.from_numpy(stitch(logits, positions, height, width)).softmax(0)
    return 1 - probs[0].numpy()


def render(
    image: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    truth: np.ndarray | None,
) -> np.ndarray:
    """Lay out the photo, the defect heatmap, and the predicted pixels side by side.

    Args:
        image: RGB photo.
        prob: Defect probability per pixel.
        pred: True where the pixel was called a defect.
        truth: 0/1 ground-truth mask, or None. When given, a fourth panel shows
            hits in green, misses in blue, and false alarms in red.

    Returns:
        One BGR picture, ready for cv2.imwrite.
    """
    photo = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = photo.copy()
    overlay[pred] = (0.4 * overlay[pred] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    panels = [photo, heat, overlay]
    if truth is not None:
        diff = photo.copy() // 2
        diff[pred & (truth == 1)] = (0, 255, 0)
        diff[~pred & (truth == 1)] = (255, 0, 0)
        diff[pred & (truth == 0)] = (0, 0, 255)
        panels.append(diff)
    return np.hstack(panels)


def main() -> None:
    """Segment each photo given and write <stem>_seg.png next to --out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("results/pvels_predefined/best.pt")
    )
    parser.add_argument("--mask", type=Path, nargs="*", help="ground truth, in order")
    parser.add_argument("--out", type=Path, default=Path("predictions"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--threshold", type=float, help="defect cut-off; default comes from the run"
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    # Older checkpoints were saved before threshold was a setting, hence the 0.5.
    threshold = args.threshold or cfg.get("threshold", 0.5)
    print(f"[threshold] {threshold}")
    args.out.mkdir(parents=True, exist_ok=True)
    masks = args.mask or [None] * len(args.images)
    for path, mask_path in zip(args.images, masks):
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        truth = None
        if mask_path is not None:
            raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            truth = (raw > (0 if raw.max() <= 1 else cfg["mask_threshold"])).astype(
                np.uint8
            )
        prob = segment(model, image, cfg, device)
        pred = prob >= threshold
        out = args.out / f"{path.stem}_seg.png"
        cv2.imwrite(str(out), render(image, prob, pred, truth))
        cv2.imwrite(str(args.out / f"{path.stem}_mask.png"), pred.astype(np.uint8) * 255)
        print(
            f"{path.name}: {pred.sum()} defect px, {(~pred).sum()} normal px "
            f"({100 * pred.mean():.2f}% defect) -> {out}"
        )


if __name__ == "__main__":
    main()
