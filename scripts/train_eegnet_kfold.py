#!/usr/bin/env python3
"""
EEGNet 5-fold stratified CV — BCIC IV-2a within-session (Palmetto cluster version)

Same model, data loading, and training loop as train_eegnet7_cluster.py.
Evaluation protocol: StratifiedKFold(5) instead of holdout split.
Goal: produce accuracy numbers on the same footing as Riemannian MDM/TS+PCA+LDA.

Usage:
  python train_eegnet_kfold.py \
    --data_dir /scratch/$USER/BCICIV_2a_gdf \
    --out_dir  /scratch/$USER/eegnet_kfold_runs \
    --subject  A01T \
    --seed     42 \
    --num_workers 4 --pin_memory --use_amp

Outputs:
  <out_dir>/<subject>_kfold_seed<seed>/
    fold_0/ ... fold_4/     (best.pth, metrics.json, train.log per fold)
    kfold_summary.json      (mean ± std across folds)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mne
mne.set_log_level("WARNING")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold


# ─────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler(str(log_path), mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ─────────────────────────────────────────
# Data loading  (BCIC IV-2a)
# ─────────────────────────────────────────
MI_KEYS = ["769", "770", "771", "772"]   # left, right, foot, tongue

@dataclass
class LoadedData:
    X: np.ndarray   # (N, C, T)  float32
    y: np.ndarray   # (N,)       int64
    sfreq: float


def _mi_event_map(event_id_from_ann: Dict[str, int]) -> Dict[int, int]:
    available = [k for k in MI_KEYS if k in event_id_from_ann]
    if len(available) < 2:
        raise ValueError(
            f"Not enough MI annotations. Expected some of {MI_KEYS}, "
            f"got: {sorted(event_id_from_ann.keys())}"
        )
    return {event_id_from_ann[k]: i for i, k in enumerate(MI_KEYS) if k in event_id_from_ann}


def load_gdf(
    path: Path,
    tmin: float = 0.0,
    tmax: float = 4.0,
    l_freq: float = 4.0,
    h_freq: float = 38.0,
) -> LoadedData:
    raw = mne.io.read_raw_gdf(str(path), preload=True, verbose="ERROR")

    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, misc=False)
    if len(picks) == 0:
        picks = mne.pick_types(raw.info, eeg=True, stim=False)
    raw.pick([raw.ch_names[i] for i in picks])

    sfreq = float(raw.info["sfreq"])
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose="ERROR")

    events, event_id_from_ann = mne.events_from_annotations(raw, verbose="ERROR")
    desired = _mi_event_map(event_id_from_ann)

    mask = np.isin(events[:, 2], list(desired.keys()))
    events = events[mask]
    if len(events) == 0:
        raise ValueError(f"No MI events found in {path.name}")

    epochs = mne.Epochs(
        raw, events, event_id=None,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose="ERROR",
    )
    X = epochs.get_data().astype(np.float32)   # (N, C, T)
    y = np.array([desired[c] for c in events[:, 2]], dtype=np.int64)
    return LoadedData(X=X, y=y, sfreq=sfreq)


# ─────────────────────────────────────────
# Dataset / preprocessing
# ─────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = np.expand_dims(self.X[idx], 0)           # (1, C, T)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


def standardize(train_X: np.ndarray, test_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1e-6
    mean = train_X.mean(axis=(0, 2), keepdims=True)
    std  = train_X.std(axis=(0, 2),  keepdims=True) + eps
    return (train_X - mean) / std, (test_X - mean) / std


# ─────────────────────────────────────────
# Model  (standard EEGNet)
# ─────────────────────────────────────────
class EEGNet(nn.Module):
    def __init__(self, n_chans: int, n_classes: int = 4, samples: int = 1000,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 dropout: float = 0.25, kernel_length: int = 64):
        super().__init__()
        self.conv1     = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1       = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)
        self.bn2       = nn.BatchNorm2d(F1 * D)
        self.act       = nn.ELU(inplace=True)
        self.pool1     = nn.AvgPool2d((1, 4))
        self.drop1     = nn.Dropout(dropout)
        self.sep_dw    = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.sep_pw    = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3       = nn.BatchNorm2d(F2)
        self.pool2     = nn.AvgPool2d((1, 8))
        self.drop2     = nn.Dropout(dropout)

        with torch.no_grad():
            feat = self._features(torch.zeros(1, 1, n_chans, samples)).shape[1]
        self.classifier = nn.Linear(feat, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))
        x = self.drop1(self.pool1(x))
        x = self.act(self.bn3(self.sep_pw(self.sep_dw(x))))
        x = self.drop2(self.pool2(x))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._features(x))


# ─────────────────────────────────────────
# Training one fold
# ─────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss_sum += float(crit(logits, yb).item()) * yb.size(0)
        correct  += int((logits.argmax(1) == yb).sum())
        total    += int(yb.size(0))
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_fold(
    train_X: np.ndarray, train_y: np.ndarray,
    test_X:  np.ndarray, test_y:  np.ndarray,
    fold_dir: Path,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    use_amp: bool,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(fold_dir / "train.log")
    logger.info(f"Device: {device} | train={train_X.shape} test={test_X.shape}")

    train_X, test_X = standardize(train_X, test_X)

    pw = num_workers > 0
    dl_tr = DataLoader(EEGDataset(train_X, train_y), batch_size=batch_size,
                       shuffle=True,  num_workers=num_workers,
                       pin_memory=pin_memory, persistent_workers=pw)
    dl_te = DataLoader(EEGDataset(test_X, test_y),  batch_size=batch_size,
                       shuffle=False, num_workers=num_workers,
                       pin_memory=pin_memory, persistent_workers=pw)

    model = EEGNet(n_chans=train_X.shape[1], samples=train_X.shape[2]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit  = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_acc, best_epoch = -1.0, -1

    for ep in range(1, epochs + 1):
        model.train()
        total = correct = 0
        loss_sum = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = model(xb)
                loss   = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item()) * yb.size(0)
            correct  += int((logits.argmax(1) == yb).sum())
            total    += int(yb.size(0))

        tr_acc = correct / max(total, 1)
        _, te_acc = evaluate(model, dl_te, device)

        logger.info(f"Ep {ep:03d}/{epochs} | tr_acc {tr_acc:.4f} | te_acc {te_acc:.4f}")

        if te_acc > best_acc:
            best_acc   = te_acc
            best_epoch = ep
            torch.save({"model_state": model.state_dict(), "epoch": ep},
                       fold_dir / "best.pth")

    metrics = {"best_acc": best_acc, "best_epoch": best_epoch, "seed": seed}
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Done — best_acc={best_acc:.4f} @ epoch {best_epoch}")
    logger.handlers.clear()
    return metrics


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--out_dir",     required=True)
    p.add_argument("--subject",     required=True, help="e.g. A01T or A01")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--n_folds",     type=int, default=5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory",  action="store_true")
    p.add_argument("--use_amp",     action="store_true")
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--tmin",        type=float, default=0.0)
    p.add_argument("--tmax",        type=float, default=4.0)
    p.add_argument("--l_freq",      type=float, default=4.0)
    p.add_argument("--h_freq",      type=float, default=38.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    subj = args.subject if args.subject.endswith("T") else args.subject + "T"
    gdf_path = Path(args.data_dir) / f"{subj}.gdf"

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Subject={subj}  seed={args.seed}  folds={args.n_folds}")
    print(f"  GDF: {gdf_path}")

    ld = load_gdf(gdf_path, tmin=args.tmin, tmax=args.tmax,
                  l_freq=args.l_freq, h_freq=args.h_freq)
    print(f"  Loaded X={ld.X.shape}  classes={np.unique(ld.y, return_counts=True)}")

    run_dir = Path(args.out_dir) / f"{subj}_kfold_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_accs: List[float] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(ld.X, ld.y)):
        fold_dir = run_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        print(f"  Fold {fold_idx}: train={len(tr_idx)}  test={len(te_idx)}", flush=True)

        m = train_fold(
            ld.X[tr_idx], ld.y[tr_idx],
            ld.X[te_idx], ld.y[te_idx],
            fold_dir=fold_dir,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            use_amp=args.use_amp,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        fold_accs.append(m["best_acc"])
        print(f"  Fold {fold_idx} best_acc={m['best_acc']:.4f}", flush=True)

    summary = {
        "subject":   subj,
        "seed":      args.seed,
        "n_folds":   args.n_folds,
        "fold_accs": fold_accs,
        "mean_acc":  float(np.mean(fold_accs)),
        "std_acc":   float(np.std(fold_accs)),
        "config": {
            "tmin": args.tmin, "tmax": args.tmax,
            "l_freq": args.l_freq, "h_freq": args.h_freq,
            "epochs": args.epochs, "lr": args.lr,
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(run_dir / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  {subj}  mean={summary['mean_acc']*100:.1f}%  std={summary['std_acc']*100:.1f}%")
    print(f"  Saved: {run_dir / 'kfold_summary.json'}")


if __name__ == "__main__":
    main()
