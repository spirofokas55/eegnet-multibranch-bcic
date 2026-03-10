# Multi-Branch EEGNet for Motor Imagery (BCIC IV-2a)

This project explores robust EEG decoding for brain-computer interfaces using a multi-branch EEGNet architecture and large-scale multi-subject evaluation.

This repository implements a multi-branch EEGNet-style deep learning pipeline for 4-class motor imagery EEG decoding using PyTorch and MNE.

The model uses two temporal branches with long and short kernels, independent spatial projections across electrodes, and fused feature representations before classification.

The goal of this project is to study EEG decoding performance across multiple subjects using the BCI Competition IV-2a dataset.

## Overview

Motor imagery classification from EEG signals is a central problem in brain-computer interface (BCI) research. EEG signals are highly variable across subjects and sessions, which makes reliable decoding difficult.

This repository provides a reproducible deep learning pipeline for running multi-subject EEG decoding experiments using a modified EEGNet architecture.

## Model Features

- Multi-branch EEGNet-style architecture
- Two temporal convolution branches with long and short kernels
- Independent spatial filtering across electrodes
- Feature fusion before classification
- Implemented using PyTorch and MNE

## What This Repository Does

- Runs within-session evaluation for each subject file (`AxxT.gdf`)
- Supports multi-subject and multi-seed experiment sweeps
- Uses sliding-window test-time voting
- Saves per-run logs and evaluation metrics
- Writes experiment summaries to `summary.csv` and `summary.json`
- Includes label-shuffle sanity checks to confirm models are not exploiting dataset artifacts

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

## Results

The primary evaluation ran a full multi-subject experiment sweep across the BCIC IV-2a dataset.

Experiment setup:

- 8 subjects (A01, A02, A03, A05, A06, A07, A08, A09)
- 5 random seeds per subject
- 40 runs total

Across all runs, the model achieved:

**70.5% ± 14.7% mean test accuracy**

Some subjects consistently reached **80–90% accuracy**, while others proved significantly harder to decode. This variability reflects the well-known challenge of EEG heterogeneity across individuals.

## Cross-Session Evaluation

To study session drift, the model was trained on the BCIC IV-2a training session (`AxxT`) and evaluated on the separate evaluation session (`AxxE`).

Across 8 subjects and 5 random seeds per subject, average performance dropped from about **79% within-session accuracy** to about **70% cross-session accuracy**.

This drop was not uniform across subjects. Some subjects transferred relatively well across sessions, while others showed much larger degradation.

Example subject-level averages:

| Subject | Within-Session Accuracy | Cross-Session Accuracy |
|---------|-------------------------|------------------------|
| A03     | 93%                     | 84%                    |
| A07     | 92%                     | 76%                    |
| A05     | 79%                     | 63%                    |
| A02     | 58%                     | 50%                    |

These results highlight a central challenge in EEG-based brain-computer interfaces: **session-to-session distribution shift**. Even when the same subject performs the same motor imagery task, signal characteristics can change enough to reduce decoding performance.

This finding shifted the focus of the project beyond benchmark accuracy and toward building models that are more robust to non-stationary neural data.

## Future Directions

Possible extensions of this project include:

- Cross-session generalization
- Cross-subject transfer learning
- Domain adaptation techniques
- Improved robustness to session drift
- Subject-independent EEG decoding pipelines
