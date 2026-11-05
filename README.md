# PV Cell Anomaly Detection

Finds defects in electroluminescence (EL) photos of solar cells, marking which
pixels are faulty. See [APPROACH.md](APPROACH.md) for why it is built this way.

## Setup

```bash
uv venv --python 3.13 --allow-existing
uv sync --extra dev
uv run pytest -q
```

PyTorch installs the CPU-only build by default. For a GPU, pull the CUDA 12.6
build instead:

```bash
uv sync --extra dev --no-group cpu --group gpu
```

## Data

Two datasets are supported. Both live under `data/`, which git ignores.

### PVEL-S (binary: defect or not)

[PVEL-S](https://www.kaggle.com/datasets/yaozhang01182010/dataset-of-solar-cells-defect-segmentation)
— 1,200 EL images, already split 840 train / 360 test. Extract to
`data/pv_kaggle/PVEL_S`:

```
PVEL_S/{train,test}/defect/0000.jpg        the photo
PVEL_S/{train,test}/label/0000_mask.jpg    where the defect is
```

Masks are JPEG, so their edges are smudged and get cut at 127 rather than 0.
Sizes are mixed, 1024px and 512px. Every image has a defect — there are no
clean cells. The 840 training images lose 15% to a validation group, so
training sees 714.

### Benchmark EL (multi-class: 29 classes)

[BenchmarkELimages](https://github.com/TheMakiran/BenchmarkELimages) (Pratt et
al., 2023) — 512px cell images with a class index painted on every pixel: 13
cell features (busbars, ribbons, border, …) and 16 defect types (crack,
inactive, corrosion, …). The class list is in
`ListOfClassesAndColorCodes_20221008.csv` inside the zip.

Use the `dataset_20221008` folder, and leave out the `flip_`, `mirror_` and
`rotate_` copies in its train folder. Training already flips and turns images
on the fly, and the shipped copies would let one cell appear four times.

```bash
unzip -q data/BenchmarkELimages-main.zip -d /tmp/bel
src=/tmp/bel/BenchmarkELimages-main/dataset_20221008
for split in train val test; do
  mkdir -p data/benchmark_el/$split/{images,masks}
  for f in $src/el_images_$split/*.png; do
    name=$(basename "$f")
    case $name in flip_*|mirror_*|rotate_*) continue ;; esac
    cp "$f" data/benchmark_el/$split/images/
    cp "$src/el_masks_$split/$name" data/benchmark_el/$split/masks/
  done
done
```

That gives 553 train / 70 val / 72 test, laid out as:

```
benchmark_el/{train,val,test}/images/ARTS_00001_r4_c2.png   the photo
benchmark_el/{train,val,test}/masks/ARTS_00001_r4_c2.png    class index per pixel
```

The dataset ships its own val group, so it is used as is and `val_frac` is
ignored.

## Training

```bash
# Full run on the shipped PVEL-S 840/360 split
uv run python -m pv_anomaly.train --config configs/pvels.yaml

# Multi-class run on Benchmark EL
uv run python -m pv_anomaly.train --config configs/benchmark_el.yaml

# Quick check on 60 images, CPU only
uv run python -m pv_anomaly.train --config configs/cpu_smoke.yaml

# Pick up an interrupted run
uv run python -m pv_anomaly.train --config configs/pvels.yaml --resume

# Five folds, for error bars (needs split_mode: cv)
for f in 0 1 2 3 4; do
  uv run python -m pv_anomaly.train --config configs/pvels.yaml --fold $f
done
```

`--fold`, `--epochs`, `--device` and `--resume` override the config file.
Everything else is set in YAML; defaults are in `src/pv_anomaly/config.py`.

Each run writes to `runs/<name>_<split>/`:

| File | What it holds |
|---|---|
| `config.yaml` | every setting the run used |
| `split.json` | which images went to train, val and test |
| `history.json` | train and val loss per epoch, rewritten every epoch |
| `metrics.json` | test scores |
| `best.pt` | the model with the lowest val loss — the one that gets scored |
| `last.pt` | the latest epoch plus optimizer state, for `--resume` |

### Settings worth knowing

| Setting | Default | What it does |
|---|---|---|
| `tile` / `stride` | 512 / 256 | tile size and step; half-overlapping tiles |
| `num_classes` | 2 | 2 for PVEL-S, 29 for Benchmark EL |
| `defect_classes` | `[]` | which class indices are defects; empty means every class but 0 |
| `threshold` | 0.5 | defect cut-off for two-class runs; pick it on val, never on test |
| `defect_frac` | 0.5 | share of each batch drawn from tiles that hold a defect |
| `alpha` | 0.5 | loss mix: `alpha` × weighted cross-entropy + (1 − `alpha`) × Dice |
| `split_mode` | `predefined` | `predefined` uses the shipped folders; `cv` pools and folds |
| `patience` | 10 | stop after this many epochs without a better val loss |

## Inference

Run a trained two-class model on any EL photo and see which pixels it calls
defect and which normal. No mask is needed.

```bash
# One photo
uv run python -m pv_anomaly.inference path/to/cell.jpg \
    --checkpoint kaggle/results2/pvels_predefined/best.pt

# Several photos, each compared with its true mask (give masks in the same order)
uv run python -m pv_anomaly.inference \
    data/pv_kaggle/PVEL_S/test/defect/0840.jpg data/pv_kaggle/PVEL_S/test/defect/0879.jpg \
    --mask data/pv_kaggle/PVEL_S/test/label/0840_mask.jpg data/pv_kaggle/PVEL_S/test/label/0879_mask.jpg \
    --checkpoint kaggle/results2/pvels_predefined/best.pt
```

| Option | Default | What it does |
|---|---|---|
| `images` | — | one or more photos |
| `--checkpoint` | `results/pvels_predefined/best.pt` | the `best.pt` to load; it carries its own settings |
| `--mask` | none | true masks, one per photo, to add a comparison panel |
| `--out` | `predictions/` | where pictures are written |
| `--threshold` | the run's `threshold`, else 0.5 | defect cut-off |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

For each photo it prints the defect and normal pixel counts and writes two
files to `--out`:

- **`<name>_mask.png`** — the answer: white is defect, black is normal, same
  size as the photo.
- **`<name>_seg.png`** — panels side by side:
  1. the photo
  2. defect probability as a heatmap, blue = sure it is normal, red = sure it
     is a defect
  3. the photo with predicted defect pixels shaded red
  4. with `--mask` only: green = defect found, blue = defect missed, red =
     false alarm, grey = both say normal

It uses the same tiling and stitching as test scoring, so what you see matches
`metrics.json`. A 1024px photo takes a few seconds on CPU.

Only two-class models work here. With a multi-class model the heatmap counts
every non-background class, so busbars and ribbons would show up as defects.

**Slow on CPU?** Training leaves millions of weights as tiny "denormal"
floats. GPUs do not care, but on a CPU they made one tile take 40 s instead of
0.3 s. `load_model` zeroes them on load; anything else that runs a checkpoint
on CPU needs the same fix.

## Scores

Per-class IoU and F1, pixel AUPRC and AUROC, and how many defects were found
grouped by defect size. Overall accuracy is left out on purpose: background is
about 97% of pixels, so a model that finds nothing still scores 97%.

**Aim for 95.5% IoU** — what Ramadan et al. report on PVEL-S with this same
split ([Sci Rep 2026](https://doi.org/10.1038/s41598-026-57859-7)). Land far
below it and check the data pipeline before changing the model.

The size breakdown is the most useful number: big defects are easy, and the
ones under about 600 pixels are the hard part.

### Results so far (PVEL-S, shipped split, U-Net + ResNet-34)

All three were trained on Kaggle GPUs; their folders are under `kaggle/`.

| Run | Epochs run | Mean IoU | Defect IoU | Defect F1 | AUPRC |
|---|---|---|---|---|---|
| `results1` — 50-epoch cap | 50 | 94.21% | 90.40% | 94.96% | 0.9918 |
| `results2` — 120-epoch cap, stopped early | 71 | 94.24% | 90.44% | 94.98% | 0.9922 |
| `pvels_lovasz` — Lovász loss in place of Dice | 84 | 94.20% | 90.38% | 94.95% | 0.9925 |

Share of defects found, by defect size in pixels:

| Run | < 100 (n=8) | 100–600 (n=161) | 600–2k (n=371) | 2k–10k (n=200) | > 10k (n=180) |
|---|---|---|---|---|---|
| `results1` | 12.5% | 36.0% | 73.9% | 76.5% | 97.2% |
| `results2` | 25.0% | 39.8% | 69.8% | 77.0% | 97.8% |
| `pvels_lovasz` | 25.0% | 42.9% | 67.7% | 77.0% | 97.8% |

What this says:

- All three land within 0.05 points of each other on IoU, about 1.3 short of
  the 95.5% target.
- Big defects are solved. Small ones are not: fewer than half of those under
  600 pixels are found.
- The size numbers swing by several points between runs that differ only in
  epoch cap, so a gain of that size from a single run is not yet evidence. Run
  the five folds before trusting one.

The Lovász run used an `overlap: lovasz` setting that is not in this
repository's code yet, so `configs/` cannot reproduce it.

## Two choices that matter

- **Tiles, not resizing.** Cracks a few pixels wide vanish if a 1024px image
  is shrunk to 224, so images are cut into overlapping 512px tiles and the
  predictions joined back up.
- **Split by image, before tiling and augmentation.** Tiles from one image
  overlap, and training images get flipped — either would leak into the test
  set if the split came later.

## Layout

```
configs/
├── pvels.yaml          full PVEL-S run
├── benchmark_el.yaml   29-class Benchmark EL run
└── cpu_smoke.yaml      2-epoch check on 60 images
src/pv_anomaly/
├── config.py           settings, defaults here and overrides in YAML
├── data.py             finding images, tiling, splitting, the Dataset
├── losses.py           weighted cross-entropy + Dice
├── metrics.py          IoU, F1, AUPRC, defect-size breakdown
├── train.py            training loop, test scoring, command line
└── inference.py        run a trained model on photos, save masks and pictures
tests/                  data and model checks
```
