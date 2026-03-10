# Multi-Branch EEGNet for Motor Imagery (BCIC IV-2a)

This repository implements a multi-branch EEGNet-style deep learning pipeline for 4-class motor imagery EEG decoding using PyTorch and MNE.

The model uses two temporal branches with different kernel lengths, independent spatial projections across electrodes, and fused feature representations before classification. The goal is to study robust EEG decoding behavior across subjects using the BCI Competition IV-2a dataset.

## Overview

This project focuses on motor imagery classification from EEG signals, a core problem in brain-computer interface (BCI) research. The repository includes a multi-subject evaluation pipeline, multi-seed experimentation, sliding-window test-time voting, and automated run logging for reproducible experiments.

## Model Features

- Multi-branch EEGNet-style architecture
- Two temporal branches with different kernel lengths
- Independent spatial filtering across electrodes
- Feature fusion before classification
- Built with PyTorch and MNE

## What This Repository Does

- Runs within-session evaluation for each subject file (`AxxT.gdf`)
- Supports multi-subject and multi-seed experiment sweeps
- Applies sliding-window test-time voting
- Saves per-run logs and evaluation metrics
- Writes aggregated experiment outputs to `summary.csv` and `summary.json`

## Dataset

This project uses the **BCI Competition IV-2a** dataset.

GDF files are **not included** in this repository. Download the dataset separately from PhysioNet or the original BCI Competition source.

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate neuro
