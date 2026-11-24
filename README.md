# PV Cell Anomaly Detection

Finds defects in electroluminescence (EL) images of solar cells and marks which
pixels are faulty. It supports binary segmentation on PVEL-S (defect or not)
and 29-class segmentation on Benchmark EL.

## Results

### Benchmark EL (29 classes)

Test split of 72 images, scored over the 23 classes that have test pixels.
The reference model is a hybrid CNN–Transformer
([*Scientific Reports*, 2026](https://doi.org/10.1038/s41598-026-57859-7));
its values are its per-class F1 converted to IoU.

| | U-Net + ResNet-34 (4 runs) | DeepLabv3+ (1 run) | reference model (1 run) |
|---|---|---|---|
| mean IoU, 23 classes | **57.61 ± 0.47** | 58.29 | 53.55 |
| defect mean IoU, 12 classes | 38.32 ± 1.18 | 40.21 | 37.98 |
| pixel-weighted IoU | 91.26 ± 0.17 | 90.99 | 91.20 |

DeepLabv3+ and U-Net differ by less than run-to-run noise.

### PVEL-S (binary)

Test split of 360 images, U-Net + ResNet-34:

| | defect IoU |
|---|---|
| no class weights, threshold 0.5 (3 runs) | 91.49 ± 0.45 |
| class weights, threshold 0.90 chosen on val (1 run) | 92.01 |
| reference model (1 run, 160 × 160, no separate test split) | 93.89 |

Full results and analysis are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Setup

```bash
uv venv --python 3.13 --allow-existing
uv sync --extra dev
uv run pytest -q
```

This installs the CPU build of PyTorch. For a GPU, use the CUDA 12.6 build:

```bash
uv sync --extra dev --no-group cpu --group gpu
```

## Data

Both datasets go under `data/`, which git ignores.

### PVEL-S

[PVEL-S](https://www.kaggle.com/datasets/yaozhang01182010/dataset-of-solar-cells-defect-segmentation):
1,200 EL images, split 840 train / 360 test. Extract to `data/pv_kaggle/PVEL_S`:

```
PVEL_S/{train,test}/defect/0000.jpg        image
PVEL_S/{train,test}/label/0000_mask.jpg    defect mask
```

- Masks are JPEG, so they are thresholded at 127.
- Images are 1024 px or 512 px.
- Every image contains a defect.
- 15% of the training images are held back for validation, so training uses 714.

### Benchmark EL

[BenchmarkELimages](https://github.com/TheMakiran/BenchmarkELimages) (Pratt et
al., 2023): 512 px cell images where each pixel holds a class index, 13 cell
features and 16 defect types. Class names are in
`ListOfClassesAndColorCodes_20221008.csv` inside the zip.

Use the `dataset_20221008` folder, and skip the `flip_`, `mirror_` and
`rotate_` copies in its train folder. Training already flips and rotates
images, and those copies would put each cell in the training set four times.

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

This gives 553 train / 70 val / 72 test images:

```
benchmark_el/{train,val,test}/images/ARTS_00001_r4_c2.png   image
benchmark_el/{train,val,test}/masks/ARTS_00001_r4_c2.png    class index per pixel
```

## Training

```bash
uv run python -m pv_anomaly.train --config configs/benchmark_el.yaml
```

Command-line options `--fold`, `--epochs`, `--device`, `--resume` and `--train-seed` override
the config file. Everything else is set in YAML, with defaults in
`src/pv_anomaly/config.py`.

| Config | What it runs |
|---|---|
| `pvels.yaml` | PVEL-S, with class weights |
| `pvels_nowt.yaml` | PVEL-S, no class weights |
| `pvels_160.yaml` | PVEL-S resized to 160 × 160 |
| `benchmark_el.yaml` | **Benchmark EL baseline**: U-Net, 256 px tiles, no class weights |
| `benchmark_el_weighted.yaml` | baseline + median-frequency class weights |
| `benchmark_el_tile512.yaml` | baseline with 512 px tiles |
| `benchmark_el_stride256.yaml` | baseline with no tile overlap |
| `benchmark_el_long.yaml` | baseline trained up to 250 epochs |
| `benchmark_el_short.yaml` | baseline cut to 15 epochs (training-budget control) |
| `benchmark_el_160.yaml` | resized to 160 × 160, epoch-based patience (stops too early) |
| `benchmark_el_160_long.yaml` | resized to 160 × 160, patience matched in steps |
| `benchmark_el_dlv3p.yaml` | baseline with DeepLabv3+ instead of U-Net |
| `benchmark_el_dlv3p_weighted.yaml` | DeepLabv3+ with median-frequency weights |
| `benchmark_el_cldice.yaml` | baseline + clDice on crack and gridline, mixed 50/50 with Dice |
| `benchmark_el_cldice_add.yaml` | baseline + 0.2 × clDice on crack, added on top of Dice |
| `benchmark_el_cldice_add_long.yaml` | the clDice run above, up to 250 epochs |
| `cpu_smoke.yaml` | 2 epochs on 60 images, for a quick check |

The class-balanced, inverse-frequency and recall-loss schemes are on the
`class-weighting-ablation` branch.

Each run writes to `runs/<name>_<split>/`:

| File | Contents |
|---|---|
| `config.yaml` | every setting the run used |
| `split.json` | which images went to train, val and test |
| `history.json` | train and val loss per epoch |
| `metrics.json` | test scores |
| `best.pt` | the model with the lowest val loss; this is the one scored |
| `last.pt` | latest epoch and optimizer state, for `--resume` |

### Main settings

| Setting | Default | What it does |
|---|---|---|
| `num_classes` | 2 | 2 for PVEL-S, 29 for Benchmark EL |
| `defect_classes` | `[]` | class indices that count as defects; empty means all but 0 |
| `tile` / `stride` | 512 / 256 | tile size and step between tiles |
| `resize` | none | resize whole images to this size first; none keeps full resolution |
| `arch` | `unet` | any `segmentation_models_pytorch` architecture, e.g. `deeplabv3plus` |
| `encoder` | `resnet34` | encoder network, ImageNet-pretrained |
| `decoder_attention` | none | `scse` adds attention to each U-Net decoder block |
| `class_weights` | true | median-frequency weights in the cross-entropy; false weights all classes equally |
| `alpha` | 0.5 | loss = `alpha` × cross-entropy + (1 − `alpha`) × Dice |
| `cldice_classes` | `[]` | thin classes also scored by clDice, e.g. `[14]` for crack; empty is off |
| `cldice_weight` | 0 | how much clDice counts; 0 is off |
| `cldice_add` | false | true adds clDice on top of Dice; false splits the Dice share with it |
| `defect_frac` | 0.5 | share of each batch taken from tiles that contain a defect |
| `threshold` | 0.5 | defect cut-off for 2-class runs; choose it on val, never on test |
| `train_seed` | none | seeds weights, sampling and augmentation; same seed repeats a run exactly |
| `split_mode` | `predefined` | `predefined` uses the dataset's folders; `cv` pools and folds |
| `patience` | 10 | stop after this many epochs without a better val loss |

`patience` and the learning-rate schedule count epochs. A change to `tile`,
`stride`, `resize` or `batch_size` changes the steps per epoch, so scale
`patience` with it (§10.8).

## Inference

Runs a trained 2-class model on any EL image. No mask is needed.

```bash
uv run python -m pv_anomaly.inference path/to/cell.jpg \
    --checkpoint kaggle/results2/pvels_predefined/best.pt

# with true masks, in the same order as the images
uv run python -m pv_anomaly.inference img1.jpg img2.jpg \
    --mask img1_mask.jpg img2_mask.jpg \
    --checkpoint kaggle/results2/pvels_predefined/best.pt
```

| Option | Default | What it does |
|---|---|---|
| `--checkpoint` | `results/pvels_predefined/best.pt` | model to load; it carries its own settings |
| `--mask` | none | true masks, adds a comparison panel |
| `--out` | `predictions/` | output folder |
| `--threshold` | the run's value, else 0.5 | defect cut-off |
| `--device` | `auto` | `cpu`, `cuda` or `auto` |

For each image it writes:

- `<name>_mask.png`: white for defect, black for normal.
- `<name>_seg.png`: the image, a defect-probability heatmap, the image with
  defects shaded red, and with `--mask`, a panel where green is found, blue is
  missed and red is a false alarm.

It uses the same tiling as test scoring, so the pictures match
`metrics.json`. Multi-class models are not supported here: the heatmap would
count busbars and ribbons as defects.

On CPU, `load_model` zeroes tiny "denormal" weights, which otherwise make
inference about 100× slower.

## Scores

`metrics.json` holds per-class IoU and F1, mean IoU, defect mean IoU, pixel
AUPRC and AUROC, and the share of defect regions found by size. Mean IoU
averages only classes that have test pixels. Pixel accuracy is left out,
because background fills most of every image.

## Layout

```
configs/                 one YAML file per experiment (see Training)
src/pv_anomaly/
├── config.py            settings and their defaults
├── data.py              finding images, tiling, splitting, the Dataset
├── losses.py            cross-entropy + Dice, class weights
├── metrics.py           IoU, F1, AUPRC, defect-size breakdown
├── train.py             training loop, test scoring, command line
└── inference.py         run a trained model on images
scripts/rescore.py       score old runs again with the current metrics
tests/                   data, model, seed and rescore checks
kaggle/                  result archives from Kaggle GPU runs
```
