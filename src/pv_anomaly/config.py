"""Run settings, with defaults here and overrides in a YAML file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    """Every setting for one run. Defaults here, overrides from YAML."""

    name: str = "pvels"
    out_dir: Path = Path("runs")
    device: str = "auto"

    # data
    root: Path = Path("data/pv_kaggle/PVEL_S")
    splits: tuple[str, ...] = ("train", "test")
    image_dir: str = "defect"
    mask_dir: str = "label"
    image_glob: str = "*.jpg"
    mask_suffix: str = "_mask"  # images 0000.jpg -> masks 0000_mask.jpg
    mask_threshold: int = 127
    mask_cache: int = 1024  # masks held in RAM, ~1 MB each; bounds growth
    num_classes: int = 2
    # Which class indices are defects, for the defect-tile sampler and the
    # size-recall report. Empty means everything above background counts.
    defect_classes: tuple[int, ...] = ()
    max_images: int | None = None  # debug aid; None uses all

    # Tile at native resolution instead of resizing: PVEL-S microcracks are a
    # few pixels wide and do not survive a resize to 224.
    tile: int = 512
    stride: int = 256
    # Resize whole images to this square size before tiling, for reproducing a
    # paper that trains at a fixed small input: Ramadan et al. use 160x160.
    # None keeps native resolution, which every result so far uses. Set tile
    # and stride to the same value so each image becomes one exact tile.
    resize: int | None = None

    # split: "predefined" honours the shipped train/test dirs (comparable with
    # published numbers); "cv" pools everything into k folds (error bars).
    split_mode: str = "predefined"
    n_folds: int = 5
    fold: int = 0
    val_frac: float = 0.15
    seed: int = 1337

    # model
    arch: str = "unet"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    # Defect cut-off for 2-class runs, picked on val. 0.5 is the same as argmax.
    threshold: float = 0.5

    # optimisation
    alpha: float = 0.5  # loss = alpha * weighted CE + (1 - alpha) * Dice
    # Median-frequency weights put background ~400x below the median class,
    # which on a 29-class run leaves no reason to get background right. False
    # weights every class equally.
    class_weights: bool = True
    # Which scheme to use when class_weights is on. "median" reproduces every
    # run before 2026-11-13. "balanced" is Cui et al.'s class-balanced loss
    # (CVPR 2019), whose beta bounds the weight ratio at ~1/(1-beta) -- the
    # published answer to median frequency's unbounded floor. "inverse" is
    # plain inverse frequency. "recall" is Tian et al.'s recall loss, which is
    # dynamic and recomputed per batch inside CEDiceLoss.
    weighting: str = "median"
    # Beta for weighting="balanced". It sets the weight spread at ~1/(1-beta),
    # measured on benchmark_el's train split: 0.99 -> 1.8:1, 0.999 -> 13.3:1,
    # 0.9999 -> 129:1, 0.99999 -> 1283:1, 0.9999999 -> 128,195:1. Median
    # frequency sits at 7735:1, so beta above ~0.999999 is WORSE than the
    # scheme it is meant to fix. Default 0.999 is the usefully bounded regime.
    #
    # The scale is dataset-specific and the difference is enormous: on binary
    # PVEL-S both classes hold ~1e8 pixels, so every beta up to 0.9999999
    # returns uniform weights and only 1-1e-9 differentiates at all. Cui et al.
    # tuned beta for SAMPLE counts in the hundreds; on pixel counts the useful
    # range moves by six orders of magnitude between datasets. Sweep it, and
    # read the spread the training log prints rather than trusting a default.
    cb_beta: float = 0.999
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    patience: int = 10
    resume: bool = False  # pick up from last.pt if the run was interrupted
    defect_frac: float = 0.5  # share of each batch drawn from defect tiles

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Read a YAML file, filling in any setting it leaves out.

        Args:
            path: The YAML file to read.

        Returns:
            A Config with the file's values applied over the defaults.

        Raises:
            ValueError: If the file has a key that is not a setting, which
                catches typos like `epocs` instead of `epochs`.
        """
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**raw)
        cfg.root, cfg.out_dir = Path(cfg.root), Path(cfg.out_dir)
        cfg.splits = tuple(cfg.splits)
        cfg.defect_classes = tuple(cfg.defect_classes)
        return cfg

    def as_dict(self) -> dict:
        """Return every setting as plain values, ready for YAML."""
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }

    def save(self, path: Path) -> None:
        """Write every resolved setting to `path`, as a record of this run."""
        path.write_text(yaml.safe_dump(self.as_dict(), sort_keys=False))

    @property
    def run_dir(self) -> Path:
        """Return the folder this run writes its results to."""
        tag = "predefined" if self.split_mode == "predefined" else f"fold{self.fold}"
        return self.out_dir / f"{self.name}_{tag}"
