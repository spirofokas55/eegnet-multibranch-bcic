# Multi-Branch EEGNet (BCIC IV-2a)

Multi-branch EEGNet-style model for 4-class motor imagery decoding using PyTorch + MNE.

## What this does
- Runs within-session evaluation per subject file (AxxT.gdf)
- Multi-subject and multi-seed sweep
- Saves per-run logs + metrics
- Writes summary.csv and summary.json

## Dataset
BCI Competition IV-2a (GDF files not included in this repo).

## Setup
Install deps:
pip install -r requirements.txt

Set dataset directory (recommended):
BCICIV2A_DIR=/path/to/BCICIV_2a_gdf

## Run
python scripts/eegnet7_multisubject.py

## Outputs
runs/
  eegnet7_A01T_seed42_YYYYmmdd_HHMMSS/
    best.pth
    metrics.json
    confusion_matrix.txt
    classification_report.txt
    train.log

and:
runs/summary.csv
runs/summary.json
