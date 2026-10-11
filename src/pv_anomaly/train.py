"""Train a defect segmenter on one split, then score it."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from pv_anomaly.config import Config
from pv_anomaly.data import (
    Tiles,
    crop,
    index_dataset,
    make_split,
    stitch,
    tile_positions,
)
from pv_anomaly.losses import CEDiceLoss, median_frequency_weights
from pv_anomaly.metrics import evaluate, report


def resolve_device(name: str) -> torch.device:
    """Pick the device to run on.

    Args:
        name: "cpu", "cuda", or "auto" to use a GPU when one is present.

    Returns:
        The torch device.
    """
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def subsample(records, cfg):
    """used only for quick experiments, to reduce the number of images used.
    Cut the image list down for quick runs, keeping every split present.

    Args:
        records: All image/mask pairs.
        cfg: Run configuration; `max_images` sets the cap, None means no cap.

    Returns:
        The trimmed list. Records arrive train-first, so this takes a share from
        each split rather than the first N overall, which would leave a
        predefined split with no test side.
    """
    if not cfg.max_images:
        return records
    per_split = max(1, cfg.max_images // len(cfg.splits))
    kept = []
    for split in cfg.splits:
        in_split = [record for record in records if record.split == split]
        kept.extend(in_split[:per_split])
    return kept


def defect_sampler(dataset: Tiles, frac: float) -> WeightedRandomSampler | None:
    """Build a sampler that draws defect tiles more often.

    Most tiles of a cell hold no defect, so picking tiles evenly would spend the
    epoch on empty background. This changes what the model sees; the loss weights
    change how it is penalised. Both are needed.

    Args:
        dataset: The training tiles.
        frac: Share of each batch to draw from defect tiles.

    Returns:
        A sampler, or None when there is nothing to rebalance.
    """
    n_defect = len(dataset.defect)
    if not frac or not n_defect or n_defect == len(dataset):
        return None
    weights = np.full(len(dataset), (1 - frac) / (len(dataset) - n_defect))
    weights[dataset.defect] = frac / n_defect
    return WeightedRandomSampler(
        torch.from_numpy(weights), len(dataset), replacement=True
    )


def run_epoch(model, loader, loss_fn, optimizer, device, scaler, desc: str) -> float:
    """Run one pass over a loader, training if an optimizer is given.

    Args:
        model: The segmentation model.
        loader: Batches of (image, mask).
        loss_fn: The loss.
        optimizer: Optimizer to step, or None to only measure.
        device: Where to run.
        scaler: Mixed-precision scaler, or None for full precision.
        desc: Label for the progress bar.

    Returns:
        Mean loss per tile.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = tiles_seen = 0
    with torch.set_grad_enabled(training):
        for images, masks in tqdm(
            loader, desc=desc, leave=False, disable=not sys.stderr.isatty()
        ):
            images, masks = images.to(device), masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device.type, dtype=torch.float16):
                    loss = loss_fn(model(images), masks)
                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss = loss_fn(model(images), masks)
                if training:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * len(images)
            tiles_seen += len(images)
    return total_loss / max(tiles_seen, 1)


@torch.no_grad()
def predict(model, dataset: Tiles, device, batch_size: int):
    """Predict whole images, one at a time, by tiling and rejoining.

    Scores are worked out on rejoined images rather than tiles, because
    overlapping tiles would count shared pixels more than once.

    Args:
        model: The trained model.
        dataset: Images to predict.
        device: Where to run.
        batch_size: Tiles per forward pass.

    Yields:
        (target, prediction, defect score) per image.
    """
    model.eval()
    cfg = dataset.cfg
    for record in dataset.records:
        image, target = dataset.image(record), dataset.mask(record)
        height, width = target.shape
        positions = tile_positions(height, width, cfg.tile, cfg.stride)
        logits = []
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            tiles = [
                dataset.transform(
                    image=crop(image, y, x, cfg.tile),
                    mask=np.zeros((cfg.tile, cfg.tile), np.uint8),
                )["image"]
                for y, x in chunk
            ]
            batch = (
                torch.from_numpy(np.stack(tiles).transpose(0, 3, 1, 2))
                .float()
                .to(device)
            )
            logits.extend(model(batch).float().cpu().numpy())
        probs = (
            torch.from_numpy(stitch(logits, positions, height, width))
            .softmax(0)
            .numpy()
        )
        yield target, probs.argmax(0).astype(np.uint8), 1 - probs[0]


def train(cfg: Config) -> dict:
    """Train on one split and score the result.

    Writes the resolved settings, the split, the best checkpoint, and the scores
    into `cfg.run_dir`.

    Args:
        cfg: Run configuration.

    Returns:
        The test scores, as returned by `evaluate`.
    """
    device = resolve_device(cfg.device)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.run_dir / "config.yaml")
    print(f"[run] {cfg.run_dir}  device={device}")

    records = index_dataset(
        cfg.root,
        cfg.splits,
        cfg.image_dir,
        cfg.mask_dir,
        cfg.image_glob,
        cfg.mask_suffix,
    )
    records = subsample(records, cfg)
    by_id = {r.image_id: r for r in records}
    train_ids, val_ids, test_ids = make_split(
        records, cfg.split_mode, cfg.n_folds, cfg.fold, cfg.val_frac, cfg.seed
    )
    (cfg.run_dir / "split.json").write_text(
        json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}, indent=2)
    )

    sets = {
        "train": Tiles([by_id[i] for i in train_ids], cfg, train=True),
        "val": Tiles([by_id[i] for i in val_ids], cfg, train=False),
        "test": Tiles([by_id[i] for i in test_ids], cfg, train=False),
    }
    print(
        f"[data] {len(records)} images -> "
        + ", ".join(f"{k} {len(v.records)} ({len(v)} tiles)" for k, v in sets.items())
    )

    # Weights come from the training fold only; using the whole dataset would
    # leak test-set class statistics into the objective.
    weights = median_frequency_weights(sets["train"].class_pixels())
    print(f"[data] class weights {[round(w, 2) for w in weights.tolist()]}")

    model = smp.create_model(
        cfg.arch,
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        in_channels=3,
        classes=cfg.num_classes,
    ).to(device)
    loss_fn = CEDiceLoss(cfg.num_classes, weights, cfg.alpha).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg.patience // 2
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    def loader(dataset, **kwargs):
        """Build a DataLoader with this run's shared settings."""
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            **kwargs,
        )

    sampler = defect_sampler(sets["train"], cfg.defect_frac)
    train_loader = loader(
        sets["train"],
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=len(sets["train"]) >= cfg.batch_size,
    )
    val_loader = loader(sets["val"], shuffle=False)

    best_loss, best_epoch = float("inf"), 0
    checkpoint = cfg.run_dir / "best.pt"
    history = []
    for epoch in range(1, cfg.epochs + 1):
        started = time.time()
        train_loss = run_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, f"epoch {epoch}"
        )
        val_loss = run_epoch(model, val_loader, loss_fn, None, device, scaler, "val")
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{epoch:3d}/{cfg.epochs}] train {train_loss:.4f}  val {val_loss:.4f}  "
            f"lr {lr:.1e}  {time.time() - started:.0f}s"
        )
        # Written every epoch
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss, "lr": lr})
        (cfg.run_dir / "history.json").write_text(json.dumps(history, indent=2))
        # A broken loss comes out as "nan", and nan never counts as better than
        # anything. So without this we would save nothing and keep going for
        # nothing. Stop here instead and say why.
        if not math.isfinite(val_loss):
            print(f"[stop] val loss is {val_loss} -- lower lr, or set amp: false")
            break
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "config": cfg.as_dict()},
                checkpoint,
            )
        elif epoch - best_epoch >= cfg.patience:
            print(f"[stop] no improvement since epoch {best_epoch}")
            break

    if not checkpoint.exists():
        raise RuntimeError(
            f"no checkpoint written: no finite val loss in {cfg.epochs} epochs"
        )
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    metrics = evaluate(
        predict(model, sets["test"], device, cfg.batch_size), cfg.num_classes
    )
    names = ["background"] + [f"defect{i}" for i in range(1, cfg.num_classes)]
    print(f"\n=== test (best epoch {best_epoch}) ===\n{report(metrics, names)}")
    (cfg.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    """Read the config file, apply any command-line overrides, and train."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pvels.yaml"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    for key in ("fold", "epochs", "device"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    train(cfg)


if __name__ == "__main__":
    main()
