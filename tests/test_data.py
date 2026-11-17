from pathlib import Path

import numpy as np
import pytest

from pv_anomaly.data import (
    Record,
    Tiles,
    crop,
    index_dataset,
    make_split,
    stitch,
    tile_positions,
)

IDS = [f"img_{i:03d}" for i in range(50)]


# --- tiling ---------------------------------------------------------------


def test_positions_cover_the_whole_image():
    covered = np.zeros((1024, 1024), bool)
    for y, x in tile_positions(1024, 1024, 512, 256):
        covered[y : y + 512, x : x + 512] = True
    assert covered.all()


def test_last_position_snaps_flush_to_the_edge():
    assert max(y for y, _ in tile_positions(1000, 1000, 512, 256)) == 1000 - 512


def test_image_smaller_than_tile_yields_one_position():
    assert tile_positions(100, 100, 512, 256) == [(0, 0)]


def test_crop_pads_a_short_source():
    tile = crop(np.ones((100, 100), np.uint8), 0, 0, 512)
    assert (
        tile.shape == (512, 512)
        and tile[:100, :100].all()
        and not tile[100:, 100:].any()
    )


def test_stitch_roundtrips_a_known_array():
    truth = np.random.default_rng(0).normal(size=(2, 300, 300)).astype(np.float32)
    positions = tile_positions(300, 300, 128, 64)
    tiles = [truth[:, y : y + 128, x : x + 128] for y, x in positions]
    assert np.allclose(stitch(tiles, positions, 300, 300), truth, atol=1e-5)


def test_stitch_rejects_uncovered_pixels():
    with pytest.raises(RuntimeError):
        stitch([np.ones((2, 2, 2), np.float32)], [(0, 0)], 4, 4)


def test_rejects_non_positive_tile():
    with pytest.raises(ValueError):
        tile_positions(100, 100, 0, 10)


# --- indexing -------------------------------------------------------------


def test_index_pairs_suffixed_masks(dataset):
    records = index_dataset(dataset)
    assert len(records) == 9
    assert all(r.mask.stem.endswith("_mask") for r in records)


def test_ids_are_split_prefixed(dataset):
    """train/0000 and test/0000 must not collapse into one CV group."""
    ids = [r.image_id for r in index_dataset(dataset)]
    assert len(set(ids)) == len(ids) == 9
    assert {"train/0000", "test/0000"} <= set(ids)


def test_wrong_suffix_finds_nothing(dataset):
    with pytest.raises(FileNotFoundError, match="mask_suffix"):
        index_dataset(dataset, mask_suffix="")


# --- splits ---------------------------------------------------------------


def _records(ids, split="train"):
    return [Record(i, Path("image"), Path("mask"), split) for i in ids]


def test_cv_partitions_are_disjoint():
    train, val, test = make_split(_records(IDS), "cv", 5, 0, 0.15, 1337)
    assert not set(train) & set(val) and not (set(train) | set(val)) & set(test)
    assert sorted(train + val + test) == sorted(IDS)


def test_cv_tests_every_image_exactly_once():
    seen = [
        i
        for f in range(5)
        for i in make_split(_records(IDS), "cv", 5, f, 0.15, 1337)[2]
    ]
    assert sorted(seen) == sorted(IDS)


def test_cv_is_deterministic_per_seed():
    a = make_split(_records(IDS), "cv", 5, 1, 0.15, 7)
    assert a == make_split(_records(IDS), "cv", 5, 1, 0.15, 7)
    assert a != make_split(_records(IDS), "cv", 5, 1, 0.15, 8)


def test_cv_covers_a_non_divisible_count():
    ids = [f"i{i}" for i in range(23)]
    seen = [
        i for f in range(5) for i in make_split(_records(ids), "cv", 5, f, 0.15, 0)[2]
    ]
    assert sorted(seen) == sorted(ids)


def test_predefined_honours_the_shipped_partition(dataset):
    train, val, test = make_split(index_dataset(dataset), "predefined", 5, 0, 0.25, 0)
    assert len(test) == 3 and all(i.startswith("test/") for i in test)
    assert len(train) + len(val) == 6
    assert all(i.startswith("train/") for i in train + val)


def test_predefined_needs_both_sides(dataset):
    train_only = [r for r in index_dataset(dataset) if r.split == "train"]
    with pytest.raises(ValueError, match="predefined"):
        make_split(train_only, "predefined", 5, 0, 0.15, 0)


@pytest.mark.parametrize("args", [("cv", 5, 5), ("bogus", 5, 0)])
def test_rejects_bad_split_arguments(args):
    mode, n_folds, fold = args
    with pytest.raises(ValueError):
        make_split(_records(IDS), mode, n_folds, fold, 0.15, 0)


# --- masks and tiles ------------------------------------------------------


def test_jpeg_mask_binarises_cleanly(cfg, dataset):
    """Lossy JPEG masks threshold to exactly {0, 1}, not to ringing noise."""
    ds = Tiles(index_dataset(dataset), cfg, train=False)
    mask = ds[0][1].numpy()
    assert set(np.unique(mask)) <= {0, 1}
    assert 300 < mask.sum() < 500  # the 20x20 block survives, ringing does not


def test_zero_one_label_map_is_not_erased(tmp_path, cfg, write):
    """A 0/1 PNG label map must survive the JPEG threshold."""
    root = tmp_path / "labelmap"
    mask = np.zeros((64, 64), np.uint8)
    mask[10:30, 10:30] = 1
    write(root / "train" / "defect" / "a.png", np.full((64, 64), 128, np.uint8))
    write(root / "train" / "label" / "a_mask.png", mask)
    cfg.root, cfg.tile, cfg.stride = root, 64, 64
    ds = Tiles(index_dataset(root, ("train",), image_glob="*.png"), cfg, train=False)
    assert ds[0][1].sum() == 400


def test_mixed_image_sizes_index_together(tmp_path, cfg, write):
    """PVEL-S ships 1024x1024 and 512x512 in one directory."""
    root = tmp_path / "mixed"
    for name, size in [("big", 256), ("small", 128)]:
        write(
            root / "train" / "defect" / f"{name}.jpg",
            np.full((size, size), 100, np.uint8),
        )
        write(
            root / "train" / "label" / f"{name}_mask.jpg",
            np.zeros((size, size), np.uint8),
        )
    cfg.root = root
    ds = Tiles(index_dataset(root, ("train",)), cfg, train=False)
    assert len(ds.records) == 2 and len(ds.tiles) == 5  # 4 tiles + 1


def test_defect_tiles_are_tracked(cfg, dataset):
    ds = Tiles(index_dataset(dataset), cfg, train=False)
    assert len(ds.defect) == len(ds.tiles) == 9  # every fixture image has a defect


def test_item_shapes_and_dtypes(cfg, dataset):
    image, mask = Tiles(index_dataset(dataset), cfg, train=True)[0]
    assert image.shape == (3, 128, 128) and image.dtype.is_floating_point
    assert (
        mask.shape == (128, 128)
        and mask.dtype.is_signed
        and not mask.dtype.is_floating_point
    )


def test_class_pixels_counts_whole_images_not_tiles(cfg, dataset):
    counts = Tiles(index_dataset(dataset), cfg, train=False).class_pixels()
    assert counts.sum() == 9 * 128 * 128
    assert counts[1] == 9 * 400


def test_max_images_keeps_every_split(dataset):
    """Records arrive train-first; a flat head would strand the predefined split."""
    from pv_anomaly.config import Config
    from pv_anomaly.train import subsample

    records = subsample(
        index_dataset(dataset), Config(max_images=4, splits=("train", "test"))
    )
    assert {r.split for r in records} == {"train", "test"}


def test_mask_cache_is_bounded(dataset, cfg):
    """Unbounded, this is ~800 MB on PVEL-S and ~17 GB on UCF-EL."""
    cfg.mask_cache = 2
    ds = Tiles(index_dataset(dataset), cfg, train=False)
    assert len(ds._masks) <= 2
    # An evicted mask must still be readable, and identical.
    first = ds.records[0]
    assert ds.mask(first).sum() == ds.mask(ds.records[-1]).sum() > 0
