# Multi-Branch EEGNet for Motor Imagery (BCIC IV-2a)

This repository contains a multi-branch EEGNet-style architecture for
4-class motor imagery decoding using PyTorch and MNE.

The model uses two temporal branches with different kernel lengths,
independent spatial projections across electrodes, and feature fusion
before classification.

## What this does
- Runs within-session evaluation per subject file (AxxT.gdf)
- Multi-subject and multi-seed sweep
- Sliding-window test-time voting
- Saves per-run logs and metrics
- Writes summary.csv and summary.json

## Dataset
BCI Competition IV-2a (GDF files not included in this repository).
Download separately from PhysioNet or the original competition site.

## Environment

Create the conda environment:

conda env create -f environment.yml
conda activate neuro


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

runs/summary.csv
runs/summary.json


## Notes

Results vary substantially by subject due to EEG heterogeneity.
Reported metrics correspond to within-session splits and multi-seed
evaluation.
