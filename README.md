# PV Cell Anomaly Detection

Supervised defect segmentation on electroluminescence (EL) images of
photovoltaic cells.

## Approach

Fully supervised semantic segmentation, not unsupervised anomaly detection.
EL is a poor fit for the latter: polycrystalline grain boundaries are visually
indistinguishable from cracks, so the "normal" manifold is wide and
reconstruction error does not separate the two. Labels are needed to evaluate
either approach, and the public EL datasets already ship them.

Two design choices carry most of the weight:

- **Tile at native resolution, never resize.** PVEL-S is 1024x1024 and
  microcracks are a few pixels wide; a resize to 224 destroys them before the
  network sees them. Images are tiled into overlapping 512px crops and
  predictions are stitched back with overlap averaging.
- **Split on source image id, before tiling and augmentation.** Splitting tiles
  would put overlapping crops of one image on both sides of the split;
  splitting after augmentation would put a rotated copy of a training image in
  the test set.

## Setup

```bash
uv venv --python 3.13 --allow-existing
uv sync --extra dev
uv run pytest -q
```

PyTorch resolves to CPU-only wheels (see `[tool.uv.index]` in
`pyproject.toml`). Drop that section for a CUDA build.

## Data

[PVEL-S](https://www.kaggle.com/datasets/yaozhang01182010/dataset-of-solar-cells-defect-segmentation)
— 1,200 EL images, shipped 840/360, extracted to `data/pv_kaggle/PVEL_S`:

```
PVEL_S/{train,test}/defect/0000.jpg
PVEL_S/{train,test}/label/0000_mask.jpg
```

Quirks the loader handles: masks are lossy 3-channel JPEG (thresholded at 127,
not `> 0`); sizes are mixed (1,068 at 1024px, 132 at 512px); every image
contains a defect, so there are no normal samples.

## Running

```bash
# Full run, shipped 70/30 split
uv run python -m pv_anomaly.train --config configs/pvels.yaml

# 5-fold cross-validation for error bars
for f in 0 1 2 3 4; do
  uv run python -m pv_anomaly.train --config configs/pvels.yaml --fold $f
done   # requires split_mode: cv

# Fast CPU check on 60 images
uv run python -m pv_anomaly.train --config configs/cpu_smoke.yaml
```

Each run writes `config.yaml`, `split.json`, `best.pt` and `metrics.json` to
`runs/<name>_<split>/`.

The full config at 512px is a GPU job; use `max_images` to iterate on CPU.

## Metrics

Per-class IoU and F1, pixel AUPRC and AUROC, and per-region recall bucketed by
defect area. Global pixel accuracy is deliberately not reported — background is
~97% of pixels, so accuracy measures the background.

**Reference: 95.54% IoU** on PVEL-S with the same 70/30 split (Ramadan et al.,
*Sci Rep* 2026, [10.1038/s41598-026-57859-7](https://doi.org/10.1038/s41598-026-57859-7)).
Land far below that and suspect the data pipeline before the architecture.

The size sweep is the key diagnostic: it reproduces the published finding that
regions above ~10k pixels are detected reliably while regions under ~600 are
missed entirely. It tells you whether a resolution change actually bought you
the small defects it was meant to.

## Layout

```
src/pv_anomaly/
├── config.py    flat dataclass loaded from YAML
├── data.py      indexing, tiling, leak-free splits, Dataset
├── losses.py    weighted CE + Dice
├── metrics.py   per-class IoU/F1, AUPRC, defect-size sweep
└── train.py     training loop, tiled inference, CLI
```
