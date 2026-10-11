# PV Cell Anomaly Detection

Finds defects in electroluminescence (EL) photos of solar cells, marking which
pixels are faulty.

## Setup

```bash
uv venv --python 3.13 --allow-existing
uv sync --extra dev
uv run pytest -q
```

PyTorch installs the CPU-only build. For a GPU, drop the `[tool.uv.index]`
section from `pyproject.toml`.

## Data

[PVEL-S](https://www.kaggle.com/datasets/yaozhang01182010/dataset-of-solar-cells-defect-segmentation)
— 1,200 EL images, already split 840 train / 360 test. Extract to
`data/pv_kaggle/PVEL_S`:

```
PVEL_S/{train,test}/defect/0000.jpg        the photo
PVEL_S/{train,test}/label/0000_mask.jpg    where the defect is
```

Masks are JPEG, so their edges are smudged and get cut at 127 rather than 0.
Sizes are mixed, 1024px and 512px. Every image has a defect — there are no
clean cells.

## Running

```bash
# Full run on the shipped 840/360 split
uv run python -m pv_anomaly.train --config configs/pvels.yaml

# Quick check on 60 images, CPU only
uv run python -m pv_anomaly.train --config configs/cpu_smoke.yaml

# Pick up an interrupted run
uv run python -m pv_anomaly.train --config configs/pvels.yaml --resume

# Five folds, for error bars (needs split_mode: cv)
for f in 0 1 2 3 4; do
  uv run python -m pv_anomaly.train --config configs/pvels.yaml --fold $f
done
```

Each run writes to `runs/<name>_<split>/`: the settings used (`config.yaml`),
which images went where (`split.json`), the loss per epoch (`history.json`),
the scores (`metrics.json`), the model that gets scored (`best.pt`) and a
restart point (`last.pt`).

## Scores

Per-class IoU and F1, pixel AUPRC and AUROC, and how many defects were found
grouped by defect size. Overall accuracy is left out on purpose: background is
about 97% of pixels, so a model that finds nothing still scores 97%.

**Aim for 95.5% IoU** — what Ramadan et al. report on PVEL-S with this same
split ([Sci Rep 2026](https://doi.org/10.1038/s41598-026-57859-7)). Land far
below it and check the data pipeline before changing the model.

The size breakdown is the most useful number: big defects are easy, and the
ones under about 600 pixels are the hard part.

## Two choices that matter

- **Tiles, not resizing.** Cracks a few pixels wide vanish if a 1024px image
  is shrunk to 224, so images are cut into overlapping 512px tiles and the
  predictions joined back up.
- **Split by image, before tiling and augmentation.** Tiles from one image
  overlap, and training images get flipped — either would leak into the test
  set if the split came later.

## Layout

```
src/pv_anomaly/
├── config.py    settings, defaults here and overrides in YAML
├── data.py      finding images, tiling, splitting, the Dataset
├── losses.py    weighted cross-entropy + Dice
├── metrics.py   IoU, F1, AUPRC, defect-size breakdown
└── train.py     training loop, prediction, command line
```
