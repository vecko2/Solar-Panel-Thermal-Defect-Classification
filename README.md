# Solar Panel Thermal Defect Classification

I am building a reproducible image classification workflow for infrared views of solar modules. The first stage is a careful dataset audit: the model can only be useful if labels, files, and evaluation splits are trustworthy. This repository currently covers the data foundation; training and evaluation are planned for later stages.

## Dataset

I use [Raptor Maps' InfraredSolarModules dataset](https://github.com/RaptorMaps/InfraredSolarModules), which contains 20,000 grayscale JPEGs at 24 × 40 pixels. It has 12 labels: 11 anomaly types and `No-Anomaly`. The source repository provides the images and `module_metadata.json`; see [data/README.md](data/README.md) for the expected layout.

The JPEG values are not calibrated temperatures. The metadata also lacks site and inspection-session IDs, so a split by image ID cannot guarantee that related panels or neighboring frames are independent. Exact pixel duplicates are kept in the same split, but visual similarity should be reviewed before making claims about generalization to new sites.

## Set up on Windows

Use Python 3.11 from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download and extract the dataset yourself so `data/module_metadata.json` and `data/images/*.jpg` exist. The notebook will not fetch or modify the source images. If PyTorch cannot load on Windows, install a compatible PyTorch build following the [official selector](https://pytorch.org/get-started/locally/) and rerun the notebook in the new environment.

## Run the data audit

```powershell
python -m jupyter lab
```

Open `notebooks/01_eda.ipynb` and run all cells. Start Jupyter from the repository root. The notebook checks metadata and image integrity, shows class counts and percentages, displays one image per class, counts exact duplicates, and writes `results/metrics/split_manifest.csv`. The manifest records image ID, relative path, class, numeric label, pixel hash, and split.

The label order is fixed in `src/dataset.py`. Split assignment uses seed 42 and aims for 70% train, 15% validation, and 15% test while keeping identical pixels together. Conflicting-label duplicate groups are marked `excluded` and never enter an experiment split. Slight deviations in image counts are possible because duplicate groups cannot be split. Training code can read the manifest and use `SolarThermalDataset` with separate train and evaluation transforms. The current train transform only allows horizontal flips; the evaluation transform only converts the native grayscale pixels to a tensor.

## What the first audit found

All 20,000 local images passed file, grayscale, and 24 × 40 dimension checks. `No-Anomaly` has 10,000 images (50%); `Diode-Multi` has 175 (0.88%), a 57:1 difference. There are 22 additional copies of exact pixel duplicates. Six duplicate groups contain conflicting labels, so their 12 images are excluded. The remaining images split into 13,992 train, 2,998 validation, and 2,998 test images. The difference-hash scan found 641 groups of visually similar candidates, including 397 groups spanning splits. These are candidates for review, not confirmed duplicates; the split should not be treated as site-independent.

## Current scope

Week 1 establishes the loader, audit notebook, and split manifest. No model metrics or checkpoints are available yet. Future experiments should select models on validation macro-F1 and reserve test images for final evaluation.

The source images and metadata were already tracked in this checkout. Adding them to `.gitignore` prevents new files from being added automatically, but does not remove previously tracked files from Git history. I will review that separately before sharing the repository.
