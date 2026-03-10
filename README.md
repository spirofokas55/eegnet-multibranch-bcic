# Multi-Branch EEGNet for Motor Imagery (BCIC IV-2a)

This repository implements a multi-branch EEGNet-style deep learning pipeline for 4-class motor imagery EEG decoding using PyTorch and MNE.

The model uses two temporal branches with different kernel lengths, independent spatial projections across electrodes, and fused feature representations before classification.

The goal of this project is to study EEG decoding performance across multiple subjects using the BCI Competition IV-2a dataset.

## Overview

Motor imagery classification from EEG signals is a central problem in brain-computer interface (BCI) research. EEG signals are highly variable across subjects and sessions, which makes reliable decoding difficult.

This repository provides a reproducible deep learning pipeline for running multi-subject EEG decoding experiments using a modified EEGNet architecture.

## Model Features

- Multi-branch EEGNet-style architecture
- Two temporal convolution branches with different kernel lengths
- Independent spatial filtering across electrodes
- Feature fusion before classification
- Implemented using PyTorch and MNE

## What This Repository Does

- Runs within-session evaluation for each subject file (`AxxT.gdf`)
- Supports multi-subject and multi-seed experiment sweeps
- Uses sliding-window test-time voting
- Saves per-run logs and evaluation metrics
- Writes experiment summaries to `summary.csv` and `summary.json`

## Dataset

This project uses the **BCI Competition IV-2a dataset**.

The GDF files are **not included** in this repository. Download the dataset separately from PhysioNet or the original BCI Competition website.

## Environment Setup

Create the conda environment:

~~~bash
conda env create -f environment.yml
conda activate neuro
~~~

Set the dataset directory:

~~~bash
export BCICIV2A_DIR=/path/to/BCICIV_2a_gdf
~~~

## Run

Run the experiment script:

~~~bash
python scripts/eegnet7_multisubject.py
~~~

## Outputs

Example run outputs:

~~~text
runs/eegnet7_A01T_seed42_YYYYmmdd_HHMMSS/
├── best.pth
├── metrics.json
├── confusion_matrix.txt
├── classification_report.txt
└── train.log
~~~

Aggregate results are written to:

~~~text
runs/summary.csv
runs/summary.json
~~~

## Notes

Results can vary significantly across subjects due to EEG heterogeneity and the difficulty of motor imagery decoding. Metrics reported in this repository correspond to within-session splits with multi-seed evaluation.

## Future Directions

Possible extensions of this project include:

- Cross-session generalization
- Cross-subject transfer learning
- Domain adaptation techniques
- Improved robustness to session drift
- Subject-independent EEG decoding pipelines
