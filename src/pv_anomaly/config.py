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
    max_images: int | None = None  # debug aid; None uses all

    # Tile at native resolution instead of resizing: PVEL-S microcracks are a
    # few pixels wide and do not survive a resize to 224.
    tile: int = 512
    stride: int = 256

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

    # optimisation
    alpha: float = 0.5  # loss = alpha * weighted CE + (1 - alpha) * Dice
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    patience: int = 10
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
