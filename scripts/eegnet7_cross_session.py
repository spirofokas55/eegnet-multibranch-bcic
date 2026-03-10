"""
Cross-session (AxxT -> AxxE) experiment runner for a Multi-Branch EEGNet variant (BCIC IV-2a).

For each subject (A01..A09) and seed, this script:
- Trains on AxxT.gdf
- Evaluates on AxxE.gdf at cue onset code '783'
- Loads official true labels from AxxE.mat
- Saves predictions, probabilities, logs, and metrics under `runs_cross_session/`

IMPORTANT:
- This script does NOT use eval labels for any training or tuning.
- Scalers are fit only on the training split from AxxT, then applied to validation and evaluation data.
"""

import os
import sys
import json
import csv
import datetime
import logging
from pathlib import Path

import numpy as np
import mne
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score
from scipy.io import loadmat


# -----------------------------
# Configuration
# -----------------------------
class Config:
    # Repro
    seed = 42

    # Data
    tmin = -0.5
    tmax = 3.5
    freq_l = 4.0
    freq_h = 40.0
    reject_threshold = 150e-6
    n_eeg_channels_take = 22

    # Architecture
    F1 = 8
    D = 2
    kern_low = 64
    kern_high = 16
    dropout = 0.50

    # Training
    batch_size = 64
    max_lr = 3e-3
    weight_decay = 1e-3
    epochs = 80
    patience = 20
    grad_clip = 1.0

    # Augmentation
    use_mixup = True
    mixup_alpha = 0.4
    use_sign_flip = True

    # Crops
    use_crops = True
    crop_seconds = 2.0
    crops_per_epoch = 4

    # Test-time inference
    tta_voting = True
    tta_stride = 0.2

    # Loss
    label_smoothing = 0.2


cfg = Config()


# -----------------------------
# Reproducibility + device
# -----------------------------
def set_seed(seed: int) -> None:
    """Seed numpy/torch RNGs and configure deterministic CUDA behavior when available."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model
# -----------------------------
class MBEEGNet(nn.Module):
    """Multi-branch EEGNet-style model with dual temporal branches and spatial filtering."""

    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4, cfg: Config | None = None):
        super().__init__()
        if cfg is None:
            raise ValueError("cfg must be provided")

        F1 = cfg.F1
        D = cfg.D
        F2 = F1 * D

        # Low-frequency timescale branch
        self.conv1_low = nn.Conv2d(1, F1, (1, cfg.kern_low), padding=(0, cfg.kern_low // 2), bias=False)
        self.bn1_low = nn.BatchNorm2d(F1)
        self.spat1_low = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn2_low = nn.BatchNorm2d(F2)
        self.elu_low = nn.ELU()
        self.pool_low = nn.AvgPool2d((1, 4))
        self.drop_low = nn.Dropout(cfg.dropout)

        # High-frequency timescale branch
        self.conv1_high = nn.Conv2d(1, F1, (1, cfg.kern_high), padding=(0, cfg.kern_high // 2), bias=False)
        self.bn1_high = nn.BatchNorm2d(F1)
        self.spat1_high = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn2_high = nn.BatchNorm2d(F2)
        self.elu_high = nn.ELU()
        self.pool_high = nn.AvgPool2d((1, 4))
        self.drop_high = nn.Dropout(cfg.dropout)

        # Fusion block
        self.fusion_conv = nn.Conv2d(F2 * 2, F2 * 2, (1, 16), padding=(0, 8), groups=F2 * 2, bias=False)
        self.point_conv = nn.Conv2d(F2 * 2, F2 * 2, (1, 1), bias=False)
        self.bn_fuse = nn.BatchNorm2d(F2 * 2)
        self.elu_fuse = nn.ELU()
        self.pool_fuse = nn.AvgPool2d((1, 8))
        self.drop_fuse = nn.Dropout(cfg.dropout)

        # Infer classifier input size from dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)

            l = self.bn1_low(self.conv1_low(dummy))
            l = self.spat1_low(l)
            l = self.drop_low(self.pool_low(self.elu_low(self.bn2_low(l))))

            h = self.bn1_high(self.conv1_high(dummy))
            h = self.spat1_high(h)
            h = self.drop_high(self.pool_high(self.elu_high(self.bn2_high(h))))

            min_t = min(l.shape[3], h.shape[3])
            l = l[:, :, :, :min_t]
            h = h[:, :, :, :min_t]

            x = torch.cat([l, h], dim=1)
            x = self.fusion_conv(x)
            x = self.point_conv(x)
            x = self.drop_fuse(self.pool_fuse(self.elu_fuse(self.bn_fuse(x))))
            flat = x.view(1, -1).size(1)

        self.classifier = nn.Linear(flat, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = self.drop_low(self.pool_low(self.elu_low(self.bn2_low(self.spat1_low(self.bn1_low(self.conv1_low(x)))))))
        h = self.drop_high(self.pool_high(self.elu_high(self.bn2_high(self.spat1_high(self.bn1_high(self.conv1_high(x)))))))

        if l.shape[3] != h.shape[3]:
            min_t = min(l.shape[3], h.shape[3])
            l = l[:, :, :, :min_t]
            h = h[:, :, :, :min_t]

        x = torch.cat([l, h], dim=1)
        x = self.drop_fuse(self.pool_fuse(self.elu_fuse(self.bn_fuse(self.point_conv(self.fusion_conv(x))))))
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# -----------------------------
# Augmentations
# -----------------------------
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4, device: str | torch.device = "cpu"):
    """Standard mixup over a batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(device)
    mixed_x = lam * x + (1 - lam) * x[idx, :]
    return mixed_x, y, y[idx], lam


def random_crop(x: torch.Tensor, crop_len: int) -> torch.Tensor:
    """Random crop along the time axis. Expects x shaped (B,1,C,T)."""
    T = x.shape[3]
    if crop_len >= T:
        return x
    start = np.random.randint(0, T - crop_len + 1)
    return x[:, :, :, start : start + crop_len]


def sign_flip_batch(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Randomly multiply each example by -1 with probability p."""
    mask = (torch.rand(x.size(0), device=x.device) < p).float().view(-1, 1, 1, 1)
    return x * (1 - 2 * mask)


# -----------------------------
# Scaling helpers
# -----------------------------
def fit_scalers(X: np.ndarray):
    """Fit a per-channel StandardScaler on training data only."""
    return [StandardScaler().fit(X[:, c, :].reshape(-1, 1)) for c in range(X.shape[1])]


def apply_scalers(X: np.ndarray, scalers):
    """Apply fitted per-channel scalers to X with shape (N, C, T)."""
    return (
        np.array(
            [
                scalers[c].transform(X[:, c, :].reshape(-1, 1)).reshape(X.shape[0], -1)
                for c in range(X.shape[1])
            ]
        )
        .transpose(1, 0, 2)
    )


def cfg_to_dict(cfg_obj: Config) -> dict:
    """Serialize a simple config object to a dict."""
    keys = [k for k in dir(cfg_obj) if not k.startswith("__") and not callable(getattr(cfg_obj, k))]
    return {k: getattr(cfg_obj, k) for k in keys}


# -----------------------------
# Data loading
# -----------------------------
def _preprocess_raw(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    """Common EEG preprocessing for BCIC IV-2a recordings."""
    raw = raw.copy()
    raw.pick("eeg")
    try:
        raw.set_eeg_reference("average", projection=False)
    except Exception:
        pass
    raw.filter(cfg.freq_l, cfg.freq_h, fir_design="firwin")
    return raw


def load_train_T(path_T: str, cfg: Config):
    """
    Load AxxT.gdf with MI labels available (769-772 or remapped equivalents).
    Returns X, y, C, T, sfreq.
    """
    if not os.path.exists(path_T):
        raise FileNotFoundError(path_T)

    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_gdf(path_T, preload=True)
    raw = _preprocess_raw(raw, cfg)

    events, event_id_from_ann = mne.events_from_annotations(raw)
    present = set(events[:, 2].tolist())

    desired_769 = {"left": 769, "right": 770, "foot": 771, "tongue": 772}
    desired_7 = {"left": 7, "right": 8, "foot": 9, "tongue": 10}

    if all(v in present for v in desired_769.values()):
        desired = desired_769
    elif all(v in present for v in desired_7.values()):
        desired = desired_7
    else:
        raise ValueError(
            "Train file MI event codes do not match expected sets.\n"
            f"Present codes: {sorted(present)}\n"
            f"Annotation map sample: {list(event_id_from_ann.items())[:10]}"
        )

    rel_events = events[np.isin(events[:, 2], list(desired.values()))]
    epochs = mne.Epochs(
        raw,
        rel_events,
        event_id=desired,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        reject={"eeg": cfg.reject_threshold},
    )

    X = epochs.get_data(copy=True)[:, : cfg.n_eeg_channels_take, :]
    inv = {desired["left"]: 0, desired["right"]: 1, desired["foot"]: 2, desired["tongue"]: 3}
    y = np.array([inv[c] for c in epochs.events[:, -1]])
    return X, y, X.shape[1], X.shape[2], float(raw.info["sfreq"])


def load_eval_E_by_cue(path_E: str, cfg: Config):
    """
    Load AxxE.gdf and epoch trials at cue onset '783' (288 trials).
    Returns X_eval, C, T, sfreq.
    """
    if not os.path.exists(path_E):
        raise FileNotFoundError(path_E)

    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_gdf(path_E, preload=True)
    raw = _preprocess_raw(raw, cfg)

    events, event_id = mne.events_from_annotations(raw)
    if "783" not in event_id:
        raise ValueError(f"Eval file missing cue annotation '783'. event_id keys: {sorted(event_id.keys())}")

    cue_code = event_id["783"]
    cue_events = events[events[:, 2] == cue_code]
    if cue_events.shape[0] != 288:
        raise ValueError(f"Expected 288 cue events for eval, got {cue_events.shape[0]}")

    epochs = mne.Epochs(
        raw,
        cue_events,
        event_id={"cue": cue_code},
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=None,
        picks="eeg",
        preload=True,
    )

    X = epochs.get_data(copy=True)[:, : cfg.n_eeg_channels_take, :]
    return X, X.shape[1], X.shape[2], float(raw.info["sfreq"])


def load_official_eval_labels(mat_path: str) -> np.ndarray:
    """Load AxxE.mat with key 'classlabel' (values 1..4). Return labels in 0..3."""
    m = loadmat(mat_path)
    if "classlabel" not in m:
        raise KeyError(f"'classlabel' not found in {mat_path}. Keys: {[k for k in m.keys() if not k.startswith('__')]}")
    y = m["classlabel"].squeeze().astype(int)
    if y.shape[0] != 288:
        raise ValueError(f"Expected 288 labels, got {y.shape[0]}")
    if not set(np.unique(y)).issubset({1, 2, 3, 4}):
        raise ValueError(f"Unexpected label values: {np.unique(y)}")
    return (y - 1).astype(int)


# -----------------------------
# Inference helpers
# -----------------------------
def make_loader_Xonly(X: np.ndarray, batch_size: int) -> DataLoader:
    """Create a dataloader for unlabeled inputs."""
    ds = TensorDataset(torch.tensor(X[:, None, :, :], dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)


def make_loader_Xy(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a dataloader for labeled inputs."""
    ds = TensorDataset(
        torch.tensor(X[:, None, :, :], dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def predict_with_tta(model: nn.Module, xb: torch.Tensor, crop_samples: int, stride_frac: float) -> torch.Tensor:
    """
    Return summed logits (B,4) using sliding-window test-time voting over xb of shape (B,1,C,T).
    """
    B, _, _, Tm = xb.shape
    logits_sum = torch.zeros(B, 4, device=xb.device)
    stride = max(1, int(stride_frac * Tm))

    for start in range(0, max(1, Tm - crop_samples + 1), stride):
        end = start + crop_samples
        if end <= Tm:
            logits_sum += model(xb[:, :, :, start:end])

    logits_sum += model(xb[:, :, :, Tm - crop_samples : Tm])
    return logits_sum


# -----------------------------
# One subject, one seed
# -----------------------------
def run_one_cross_session(
    subject: str,
    path_T: str,
    path_E: str,
    labels_E_mat: str,
    seed: int,
    base_run_dir: Path,
) -> dict:
    """Train on AxxT and evaluate on AxxE for one subject and one seed."""
    cfg.seed = int(seed)
    set_seed(cfg.seed)

    run_dir = base_run_dir / f"crosssess_{subject}_seed{seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{subject}_seed{seed}_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(run_dir / "train.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Cross-session | Subject={subject} | Seed={seed} | Device={device}")
    logger.info(f"Train: {path_T}")
    logger.info(f"Eval : {path_E}")
    logger.info(f"Labels(E): {labels_E_mat}")

    # Load training session
    X, y, C, T, sfreq = load_train_T(path_T, cfg)
    logger.info(f"Train loaded X={X.shape} y={y.shape} sfreq={sfreq}")
    uniq, cnt = np.unique(y, return_counts=True)
    logger.info(f"Train class counts (0..3): {dict(zip(uniq.tolist(), cnt.tolist()))}")

    # Split AxxT into train/val only
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=cfg.seed
    )

    # Fit scalers on train only
    scalers = fit_scalers(X_train)
    X_train = apply_scalers(X_train, scalers)
    X_val = apply_scalers(X_val, scalers)

    train_dl = make_loader_Xy(X_train, y_train, cfg.batch_size, shuffle=True)
    val_dl = make_loader_Xy(X_val, y_val, cfg.batch_size, shuffle=False)

    crop_samples = int(cfg.crop_seconds * sfreq)
    if crop_samples <= 0:
        raise ValueError("crop_samples computed <= 0. Check crop_seconds and sfreq.")

    # Model + optimization
    model = MBEEGNet(C, crop_samples, 4, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.max_lr,
        steps_per_epoch=len(train_dl) * cfg.crops_per_epoch,
        epochs=cfg.epochs,
        pct_start=0.3,
        div_factor=25,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    best_val_acc = 0.0
    patience_ct = 0
    best_path = run_dir / "best.pth"

    logger.info("Training started (train on AxxT, validate on held-out split of AxxT)")

    for ep in range(cfg.epochs):
        model.train()
        t_loss = 0.0

        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            for _ in range(cfg.crops_per_epoch):
                x_c = random_crop(xb, crop_samples)

                if cfg.use_sign_flip:
                    x_c = sign_flip_batch(x_c, p=0.5)

                if cfg.use_mixup:
                    xm, ya, yb_mix, lam = mixup_data(x_c, yb, cfg.mixup_alpha, device)
                    out = model(xm)
                    loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb_mix)
                else:
                    out = model(x_c)
                    loss = criterion(out, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()

                t_loss += float(loss.item())

        avg_loss = t_loss / max(1, (len(train_dl) * cfg.crops_per_epoch))

        # Validation on held-out split of AxxT
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                if cfg.tta_voting:
                    logits_sum = predict_with_tta(model, xb, crop_samples, cfg.tta_stride)
                    preds = logits_sum.argmax(1)
                else:
                    preds = model(xb[:, :, :, :crop_samples]).argmax(1)

                correct += (preds == yb).sum().item()
                total += int(yb.size(0))

        val_acc = correct / max(1, total)
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(f"Ep {ep+1:03d} | Loss: {avg_loss:.4f} | Val Acc(T-split): {val_acc:.4f} | LR: {lr_now:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ct = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_ct += 1
            if patience_ct >= cfg.patience:
                logger.info("Early stopping")
                break

    # Load best checkpoint and evaluate on AxxE
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    X_eval, C2, T2, sfreq2 = load_eval_E_by_cue(path_E, cfg)
    if C2 != C:
        raise ValueError(f"Channel mismatch train C={C} vs eval C={C2}")
    X_eval = apply_scalers(X_eval, scalers)

    eval_dl = make_loader_Xonly(X_eval, cfg.batch_size)

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for (xb,) in eval_dl:
            xb = xb.to(device, non_blocking=True)

            if cfg.tta_voting:
                logits = predict_with_tta(model, xb, crop_samples, cfg.tta_stride)
            else:
                logits = model(xb[:, :, :, :crop_samples])

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())

    probs_eval = np.concatenate(all_probs, axis=0)
    preds_eval = np.concatenate(all_preds, axis=0)

    np.save(run_dir / "preds_eval.npy", preds_eval.astype(np.int64))
    np.save(run_dir / "probs_eval.npy", probs_eval.astype(np.float32))
    logger.info(f"Saved eval preds -> {run_dir / 'preds_eval.npy'} | shape={preds_eval.shape} | unique={np.unique(preds_eval)}")
    logger.info(f"Saved eval probs -> {run_dir / 'probs_eval.npy'} | shape={probs_eval.shape}")

    # Official evaluation labels
    y_true = load_official_eval_labels(labels_E_mat)
    if len(y_true) != len(preds_eval):
        raise ValueError(f"Label/pred length mismatch: y_true={len(y_true)} preds={len(preds_eval)}")

    cs_acc = float(accuracy_score(y_true, preds_eval))
    cs_kappa = float(cohen_kappa_score(y_true, preds_eval))
    cm = confusion_matrix(y_true, preds_eval)
    report = classification_report(y_true, preds_eval, target_names=["Left", "Right", "Foot", "Tongue"])

    (run_dir / "confusion_matrix_evalE.txt").write_text(str(cm), encoding="utf-8")
    (run_dir / "classification_report_evalE.txt").write_text(report, encoding="utf-8")

    metrics = {
        "subject": subject,
        "seed": int(seed),
        "train_gdf": str(path_T),
        "eval_gdf": str(path_E),
        "eval_labels_mat": str(labels_E_mat),
        "device": str(device),
        "val_best_acc_on_Tsplit": float(best_val_acc),
        "cross_session_acc_evalE": float(cs_acc),
        "cross_session_kappa_evalE": float(cs_kappa),
        "confusion_matrix_evalE": cm.tolist(),
        "timestamp": datetime.datetime.now().isoformat(),
        "cfg": cfg_to_dict(cfg),
        "note": "Cross-session evaluation: train on AxxT, eval on AxxE at cue onset '783' with official labels from .mat.",
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"FINAL (Cross-session) | {subject} seed {seed} | Acc(E): {cs_acc*100:.2f}% | Kappa(E): {cs_kappa:.4f}")
    logger.info(f"Artifacts: {run_dir}")

    logger.handlers.clear()
    return metrics


# -----------------------------
# Summary
# -----------------------------
def mean_std(x):
    """Return mean and std for a numeric iterable."""
    x = np.array(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=0))


def write_summary(base_run_dir: Path, all_metrics: list):
    """Write per-run CSV and aggregate JSON summary under the run directory."""
    base_run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_run_dir / "summary.csv"
    fieldnames = [
        "subject",
        "seed",
        "val_best_acc_on_Tsplit",
        "cross_session_acc_evalE",
        "cross_session_kappa_evalE",
        "run_dir",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in all_metrics:
            w.writerow(
                {
                    "subject": m["subject"],
                    "seed": m["seed"],
                    "val_best_acc_on_Tsplit": m["val_best_acc_on_Tsplit"],
                    "cross_session_acc_evalE": m["cross_session_acc_evalE"],
                    "cross_session_kappa_evalE": m["cross_session_kappa_evalE"],
                    "run_dir": m.get("run_dir", ""),
                }
            )

    if not all_metrics:
        summary = {
            "n_runs": 0,
            "overall_cross_session_acc_mean": None,
            "overall_cross_session_acc_std": None,
            "created_at": datetime.datetime.now().isoformat(),
        }
    else:
        subjects = sorted(set(m["subject"] for m in all_metrics))
        seeds = sorted(set(m["seed"] for m in all_metrics))

        by_subject = {s: [] for s in subjects}
        for m in all_metrics:
            by_subject[m["subject"]].append(m["cross_session_acc_evalE"])

        per_subject_mean = {s: mean_std(by_subject[s])[0] for s in subjects}
        per_subject_std = {s: mean_std(by_subject[s])[1] for s in subjects}

        overall_mean, overall_std = mean_std([m["cross_session_acc_evalE"] for m in all_metrics])

        summary = {
            "n_runs": len(all_metrics),
            "subjects": subjects,
            "seeds": seeds,
            "overall_cross_session_acc_mean": overall_mean,
            "overall_cross_session_acc_std": overall_std,
            "per_subject_cross_session_acc_mean": per_subject_mean,
            "per_subject_cross_session_acc_std": per_subject_std,
            "created_at": datetime.datetime.now().isoformat(),
        }

    with open(base_run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Runs: {len(all_metrics)}")
    if all_metrics:
        print(
            f"Cross-session Acc mean±std: "
            f"{summary['overall_cross_session_acc_mean']*100:.2f}% ± {summary['overall_cross_session_acc_std']*100:.2f}%"
        )
    print(f"Saved: {csv_path}")
    print(f"Saved: {base_run_dir / 'summary.json'}")


# -----------------------------
# Main
# -----------------------------
def main():
    """
    Expected environment variables:

    BCICIV2A_DIR:
        Directory containing A01T.gdf ... A09T.gdf and A01E.gdf ... A09E.gdf
        Default: data/BCICIV_2a_gdf

    BCICIV2A_LABELS_DIR:
        Directory containing A01E.mat ... A09E.mat
        Default: data/BCICIV_2a_labels

    RUN_ROOT:
        Output directory for cross-session artifacts
        Default: runs_cross_session
    """
    base_dir = Path(os.environ.get("BCICIV2A_DIR", "data/BCICIV_2a_gdf"))
    labels_dir = Path(os.environ.get("BCICIV2A_LABELS_DIR", "data/BCICIV_2a_labels"))

    run_root = Path(os.environ.get("RUN_ROOT", "runs_cross_session"))
    run_root.mkdir(parents=True, exist_ok=True)

    subjects = [f"A{str(i).zfill(2)}" for i in range(1, 10)]  # A01..A09; missing files are skipped automatically
    seeds = [42, 43, 44, 45, 46]

    print(f"Device: {device}")
    print(f"GDF dir: {base_dir}")
    print(f"Labels dir: {labels_dir}")
    print(f"Run outputs: {run_root.resolve()}")

    all_metrics = []

    for subj in subjects:
        path_T = base_dir / f"{subj}T.gdf"
        path_E = base_dir / f"{subj}E.gdf"
        mat_E = labels_dir / f"{subj}E.mat"

        if not path_T.exists():
            print(f"[SKIP] Missing train file: {path_T}")
            continue
        if not path_E.exists():
            print(f"[SKIP] Missing eval file: {path_E}")
            continue
        if not mat_E.exists():
            print(f"[SKIP] Missing eval labels: {mat_E}")
            continue

        for seed in seeds:
            try:
                m = run_one_cross_session(
                    subject=subj,
                    path_T=str(path_T),
                    path_E=str(path_E),
                    labels_E_mat=str(mat_E),
                    seed=seed,
                    base_run_dir=run_root,
                )
                m["run_dir"] = ""
                all_metrics.append(m)
            except Exception as e:
                print(f"[ERROR] {subj} seed {seed} failed: {e}")
                break

    write_summary(run_root, all_metrics)


if __name__ == "__main__":
    main()
