"""Find image/mask pairs, cut them into tiles, and split them safely."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Record:
    """One image file and its mask file. Paths only; nothing is read here."""

    image_id: str
    image: Path
    mask: Path
    split: str


def index_dataset(
    root: Path,
    splits: tuple[str, ...] = ("train", "test"),
    image_dir: str = "defect",
    mask_dir: str = "label",
    image_glob: str = "*.jpg",
    mask_suffix: str = "_mask",
) -> list[Record]:
    """Find every image and its mask file under <root>/<split>/<image_dir>.

    An image and a mask belong together when their names match, allowing for the
    mask's extra suffix: 0000.jpg pairs with 0000_mask.jpg.

    Args:
        root: Dataset folder.
        splits: Subfolders of `root` to search.
        image_dir: Folder holding the photos, inside each split.
        mask_dir: Folder holding the masks, inside each split.
        image_glob: Which filenames count as photos.
        mask_suffix: Extra text at the end of mask filenames.

    Returns:
        One Record per pair. Ids are prefixed with the split, so the same
        filename in train/ and test/ stays two separate images.

    Raises:
        FileNotFoundError: If no pairs are found.
    """
    records = []
    for split in splits:
        image_root, mask_root = root / split / image_dir, root / split / mask_dir
        if not image_root.is_dir():
            continue
        masks_by_stem = {
            path.stem: path
            for path in sorted(mask_root.rglob("*"))
            if path.suffix.lower() in MASK_EXTS
        }
        for image_path in sorted(image_root.rglob(image_glob)):
            mask_path = masks_by_stem.get(image_path.stem + mask_suffix)
            if mask_path is not None:
                records.append(
                    Record(f"{split}/{image_path.stem}", image_path, mask_path, split)
                )
    if not records:
        raise FileNotFoundError(
            f"no image/mask pairs under {root} "
            f"(splits={splits}, glob={image_glob!r}, mask_suffix={mask_suffix!r})"
        )
    return records


def tile_positions(
    height: int, width: int, tile: int, stride: int
) -> list[tuple[int, int]]:
    """List the top-left corner of every tile needed to cover an image.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        tile: Tile size in pixels.
        stride: Gap between tile starts. Smaller than `tile` means they overlap.

    Returns:
        (y, x) corners. The last tile on each axis is pushed flush against the
        far edge, so no part of the image is left out.

    Raises:
        ValueError: If `tile` or `stride` is not positive.
    """
    if tile <= 0 or stride <= 0:
        raise ValueError("tile and stride must be positive")

    def tile_starts(length: int) -> list[int]:
        """Return tile start positions along one axis."""
        if length <= tile:
            return [0]
        starts = list(range(0, length - tile + 1, stride))
        if starts[-1] != length - tile:
            starts.append(length - tile)
        return starts

    return [(y, x) for y in tile_starts(height) for x in tile_starts(width)]


def crop(array: np.ndarray, y: int, x: int, tile: int) -> np.ndarray:
    """Cut one tile out of an image or mask.

    Args:
        array: What to cut from.
        y: Top edge of the tile.
        x: Left edge of the tile.
        tile: Tile size in pixels.

    Returns:
        A `tile` by `tile` crop, zero-padded if the source is smaller.
    """
    out = array[y : y + tile, x : x + tile]
    pad_y, pad_x = tile - out.shape[0], tile - out.shape[1]
    if pad_y or pad_x:
        out = np.pad(out, [(0, pad_y), (0, pad_x)] + [(0, 0)] * (out.ndim - 2))
    return out


def stitch(tiles: list[np.ndarray], positions, height: int, width: int) -> np.ndarray:
    """Join tile predictions back into one full-size map.

    Args:
        tiles: Per-tile scores, each shaped (classes, tile, tile).
        positions: The (y, x) corner each tile came from.
        height: Full image height.
        width: Full image width.

    Returns:
        Scores shaped (classes, height, width). Where tiles overlap, their
        values are averaged.

    Raises:
        RuntimeError: If the tiles leave any pixel uncovered.
    """
    # total has a shape of (classes, height, width), classes are the number of classes in the segmentation task,
    # it is like channels in cnn. count has a shape of (1, height, width), it counts how many tiles cover each pixel.

    total = np.zeros((tiles[0].shape[0], height, width), np.float32)
    count = np.zeros((1, height, width), np.float32)
    for logits, (y, x) in zip(tiles, positions):
        h, w = min(logits.shape[1], height - y), min(logits.shape[2], width - x)
        total[:, y : y + h, x : x + w] += logits[:, :h, :w]
        count[:, y : y + h, x : x + w] += 1
    if not count.all():
        raise RuntimeError("tiles do not cover the image")
    return total / count


def make_split(records, mode, n_folds, fold, val_frac, seed):
    """Divide images into training, validation and test groups.

    Splits by image, never by tile, and always before augmentation. Both matter:
    tiles from one image overlap, and a flipped copy of a training image must not
    land in the test group.

    Args:
        records: All available image/mask pairs.
        mode: "predefined" uses the train/ and test/ folders as shipped. "cv"
            pools every image and uses fold `fold` as the test group.
        n_folds: How many folds to make. Used by "cv" only.
        fold: Which fold is the test group. Used by "cv" only.
        val_frac: Share of the training images kept back for validation.
        seed: Random seed, so the same split comes out every time.

    Returns:
        Three lists of image ids: (train, val, test).

    Raises:
        ValueError: If `mode` is unknown, or the fold cannot be made.
    """
    rng = np.random.default_rng(seed)

    if mode == "predefined":
        train_val = sorted({r.image_id for r in records if r.split == "train"})
        test = sorted({r.image_id for r in records if r.split == "test"})
        if not train_val or not test:
            raise ValueError(
                "predefined split needs records from both train/ and test/"
            )
    elif mode == "cv":
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in [0, {n_folds})")
        ids = sorted({r.image_id for r in records})
        rng.shuffle(ids)
        # Take every n_folds-th image, so each one is tested exactly once
        # across the full set of folds.
        test = ids[fold::n_folds]
        in_test = set(test)
        train_val = [i for i in ids if i not in in_test]
    else:
        raise ValueError(f"unknown split_mode {mode!r}; expected 'predefined' or 'cv'")

    rng.shuffle(train_val)

    n_val = round(len(train_val) * val_frac)
    val, train = train_val[:n_val], train_val[n_val:]

    assert not set(train_val) & set(test), "split leak"
    return train, val, test


def augmentations(train: bool) -> A.Compose:
    """Build the image transforms.

    Training gets flips, quarter turns, and mild brightness and noise changes.
    Only quarter turns: a free rotation blends neighbouring pixels, which blurs
    the thin cracks we are trying to keep.

    Args:
        train: True for training transforms, False to only normalise.

    Returns:
        A pipeline that takes `image` and `mask` together.
    """
    norm = A.Normalize(mean=MEAN, std=STD)
    if not train:
        return A.Compose([norm])
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
            A.GaussNoise(p=0.2),
            norm,
        ]
    )


class Tiles(Dataset):
    """A dataset of tiles cut from a fixed list of images.

    Transforms run when a tile is read, so no augmented copy is ever stored and
    none can cross a split boundary.

    Attributes:
        records: The images this dataset draws from.
        tiles: One (record index, y, x) entry per tile.
        defect: Indices of tiles holding at least one defect pixel. The training
            sampler uses these to balance batches.
    """

    def __init__(self, records: list[Record], cfg, train: bool) -> None:
        """Index every tile and note which ones contain a defect.

        Args:
            records: Image/mask pairs to draw tiles from.
            cfg: Run configuration.
            train: Whether to apply training augmentation.
        """
        self.records, self.cfg, self.train = records, cfg, train
        self.transform = augmentations(train)
        self._masks: dict[str, np.ndarray] = {}
        self.tiles: list[tuple[int, int, int]] = []
        self.defect: list[int] = []
        for i, record in enumerate(records):
            mask = self.mask(record)
            for y, x in tile_positions(*mask.shape[:2], cfg.tile, cfg.stride):
                if crop(mask, y, x, cfg.tile).any():
                    self.defect.append(len(self.tiles))
                self.tiles.append((i, y, x))

    def mask(self, record: Record) -> np.ndarray:
        """Load a mask as 0 for background and 1 for defect.

        Mask files differ between dataset copies: some store 0 and 1, others 0
        and 255 as lossy JPEG. Thresholding stops JPEG smudge around an edge from
        counting as defect. Recently used masks are cached, up to
        `cfg.mask_cache` of them.

        Args:
            record: The pair whose mask to load.

        Returns:
            A uint8 array of 0s and 1s.

        Raises:
            FileNotFoundError: If the mask file cannot be read.
        """
        if record.image_id not in self._masks:
            raw = cv2.imread(str(record.mask), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise FileNotFoundError(record.mask)
            # Encodings vary by mirror: 0/1 PNG label maps and lossy 0/255 JPEG
            # both occur. A fixed `> 0` turns JPEG ringing into defect pixels;
            # a fixed `> 127` erases a 0/1 label map. Decide per file.
            threshold = 0 if raw.max() <= 1 else self.cfg.mask_threshold
            if len(self._masks) >= self.cfg.mask_cache:
                del self._masks[next(iter(self._masks))]
            self._masks[record.image_id] = (raw > threshold).astype(np.uint8)
        return self._masks[record.image_id]

    def image(self, record: Record) -> np.ndarray:
        """Load a photo as RGB. Reads and decodes the file on every call.

        Args:
            record: The pair whose photo to load.

        Returns:
            A uint8 array shaped (height, width, 3).

        Raises:
            FileNotFoundError: If the image file cannot be read.
        """
        raw = cv2.imread(str(record.image), cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(record.image)
        # EL is single-channel; cv2 replicated it, and keeping 3 channels keeps
        # the ImageNet encoder weights meaningful.
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        """Return the number of tiles."""
        return len(self.tiles)

    def __getitem__(self, i: int):
        """Return one augmented tile.

        Args:
            i: Tile index.

        Returns:
            An (image, mask) pair of tensors, shaped (3, tile, tile) and
            (tile, tile).
        """
        record_idx, y, x = self.tiles[i]
        record = self.records[record_idx]
        tile = self.cfg.tile
        out = self.transform(
            image=crop(self.image(record), y, x, tile),
            mask=crop(self.mask(record), y, x, tile),
        )
        return (
            torch.from_numpy(out["image"].transpose(2, 0, 1).copy()).float(),
            torch.from_numpy(out["mask"].copy()).long(),
        )

    def class_pixels(self) -> np.ndarray:
        """Count pixels per class, for the loss weights.

        Counted over whole images rather than tiles, so overlapping tiles do not
        count the same pixels twice.

        Returns:
            One pixel count per class.
        """
        counts = np.zeros(self.cfg.num_classes, np.int64)
        for record in self.records:
            binned = np.bincount(
                self.mask(record).ravel(), minlength=self.cfg.num_classes
            )
            counts += binned[: self.cfg.num_classes]
        return counts
