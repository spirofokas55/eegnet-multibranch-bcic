"""
Multi-subject, multi-seed experiment runner for a Multi-Branch EEGNet variant (BCIC IV-2a).

For each subject (A01T..A09T) and seed, this script:
- Loads and band-pass filters EEG
- Epochs motor imagery trials
- Performs a within-subject train/val/test split
- Trains with optional cropping + augmentation
- Saves per-run artifacts and an aggregate summary under `runs/`
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
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score


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
    """Seed numpy/torch RNGs and configure deterministic CUDA behavior (when available)."""
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
    """Multi-branch EEGNet-style model with independent temporal branches and spatial filtering."""

    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4, cfg: Config | None = None):
        super().__init__()
        if cfg is None:
            raise ValueError("cfg must be provided")

        F1 = cfg.F1
        D = cfg.D
        F2 = F1 * D

        # Branch 1: longer temporal kernel (lower-frequency timescale)
        self.conv1_low = nn.Conv2d(1, F1, (1, cfg.kern_low), padding=(0, cfg.kern_low // 2), bias=False)
        self.bn1_low = nn.BatchNorm2d(F1)
        self.spat1_low = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn2_low = nn.BatchNorm2d(F2)
        self.elu_low = nn.ELU()
        self.pool_low = nn.AvgPool2d((1, 4))
        self.drop_low = nn.Dropout(cfg.dropout)

        # Branch 2: shorter temporal kernel (higher-frequency timescale)
        self.conv1_high = nn.Conv2d(1, F1, (1, cfg.kern_high), padding=(0, cfg.kern_high // 2), bias=False)
        self.bn1_high = nn.BatchNorm2d(F1)
        self.spat1_high = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn2_high = nn.BatchNorm2d(F2)
        self.elu_high = nn.ELU()
        self.pool_high = nn.AvgPool2d((1, 4))
        self.drop_high = nn.Dropout(cfg.dropout)

        # Fusion + classifier head
        self.fusion_conv = nn.Conv2d(F2 * 2, F2 * 2, (1, 16), padding=(0, 8), groups=F2 * 2, bias=False)
        self.point_conv = nn.Conv2d(F2 * 2, F2 * 2, (1, 1), bias=False)
        self.bn_fuse = nn.BatchNorm2d(F2 * 2)
        self.elu_fuse = nn.ELU()
        self.pool_fuse = nn.AvgPool2d((1, 8))
        self.drop_fuse = nn.Dropout(cfg.dropout)

        # Infer flatten dimension from a dummy forward pass
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
        # Low branch
        l = self.conv1_low(x)
        l = self.bn1_low(l)
        l = self.spat1_low(l)
        l = self.bn2_low(l)
        l = self.elu_low(l)
        l = self.pool_low(l)
        l = self.drop_low(l)

        # High branch
        h = self.conv1_high(x)
        h = self.bn1_high(h)
        h = self.spat1_high(h)
        h = self.bn2_high(h)
        h = self.elu_high(h)
        h = self.pool_high(h)
        h = self.drop_high(h)

        # Align time dimension if pooling/conv padding differs slightly
        if l.shape[3] != h.shape[3]:
            min_t = min(l.shape[3], h.shape[3])
            l = l[:, :, :, :min_t]
            h = h[:, :, :, :min_t]

        # Fuse
        x = torch.cat([l, h], dim=1)
        x = self.fusion_conv(x)
        x = self.point_conv(x)
        x = self.bn_fuse(x)
        x = self.elu_fuse(x)
        x = self.pool_fuse(x)
        x = self.drop_fuse(x)

        x = x.view(x.size(0), -1)
        return self.classifier(x)


# -----------------------------
# Augmentations
# -----------------------------
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4, device: str | torch.device = "cpu"):
    """Standard mixup over a batch (returns mixed_x, y_a, y_b, lambda)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(device)
    mixed_x = lam * x + (1 - lam) * x[idx, :]
    return mixed_x, y, y[idx], lam


def random_crop(x: torch.Tensor, crop_len: int) -> torch.Tensor:
    """Randomly crop along the time axis. Expects x shaped (B,1,C,T)."""
    T = x.shape[3]
    if crop_len >= T:
        return x
    start = np.random.randint(0, T - crop_len + 1)
    return x[:, :, :, start : start + crop_len]


def sign_flip_batch(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Randomly multiply each example by -1 with probability p. Expects x shaped (B,1,C,T)."""
    if not torch.is_tensor(x):
        return x
    mask = (torch.rand(x.size(0), device=x.device) < p).float().view(-1, 1, 1, 1)
    return x * (1 - 2 * mask)


# -----------------------------
# Data loading + preprocessing
# -----------------------------
def load_data(path: str, cfg: Config):
    """Load a single BCIC IV-2a .gdf file and return (X, y, C, T, sfreq)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_gdf(path, preload=True)
    raw.pick("eeg")

    # Average reference (best-effort; some files/settings may raise)
    try:
        raw.set_eeg_reference("average", projection=False)
    except Exception:
        pass

    raw.filter(cfg.freq_l, cfg.freq_h, fir_design="firwin")

    events, event_id_from_ann = mne.events_from_annotations(raw)
    present = set(events[:, 2].tolist())

    # Common BCIC IV-2a motor imagery codes
    desired_769 = {"left": 769, "right": 770, "foot": 771, "tongue": 772}
    # Alternate mapping that may appear depending on annotation remapping
    desired_7 = {"left": 7, "right": 8, "foot": 9, "tongue": 10}

    # Choose a mapping that is fully present in this file
    if all(v in present for v in desired_769.values()):
        desired = desired_769
    elif all(v in present for v in desired_7.values()):
        desired = desired_7
    else:
        raise ValueError(
            "Event codes do not match expected motor imagery label sets.\n"
            f"Present codes (unique): {sorted(present)}\n"
            f"Annotation event_id map sample: {list(event_id_from_ann.items())[:10]}"
        )

    # Keep only relevant events for epoching
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


def fit_scalers(X: np.ndarray):
    """Fit a per-channel StandardScaler on training data only."""
    return [StandardScaler().fit(X[:, c, :].reshape(-1, 1)) for c in range(X.shape[1])]


def apply_scalers(X: np.ndarray, scalers):
    """Apply per-channel scalers. Input X: (N, C, T). Output: (N, C, T)."""
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
    """Serialize a simple config object to a dict (public attributes only)."""
    keys = [k for k in dir(cfg_obj) if not k.startswith("__") and not callable(getattr(cfg_obj, k))]
    return {k: getattr(cfg_obj, k) for k in keys}


# -----------------------------
# Training + evaluation (one subject, one seed)
# -----------------------------
def run_one(subject_id: str, gdf_path: str, seed: int, base_run_dir: Path) -> dict:
    # Reproducibility for this run
    cfg.seed = int(seed)
    set_seed(cfg.seed)

    # Run output directory
    run_dir = base_run_dir / f"eegnet7_{subject_id}_seed{seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Per-run logger (file + stdout)
    logger = logging.getLogger(f"{subject_id}_seed{seed}_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(run_dir / "train.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Experiment | Subject={subject_id} | Seed={seed} | Device={device}")
    logger.info(f"Input: {gdf_path}")

    # 1) Load + preprocess
    X, y, C, T, sfreq = load_data(gdf_path, cfg)
    logger.info(f"Loaded X={X.shape} y={y.shape} sfreq={sfreq}")

    # Quick label distribution sanity check
    uniq, cnt = np.unique(y, return_counts=True)
    logger.info(f"Class counts (0=Left,1=Right,2=Foot,3=Tongue): {dict(zip(uniq.tolist(), cnt.tolist()))}")

    # 2) Split (within-subject)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=cfg.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=cfg.seed
    )

    # 3) Scale per-channel (fit on train only)
    scalers = fit_scalers(X_train)
    X_train = apply_scalers(X_train, scalers)
    X_val = apply_scalers(X_val, scalers)
    X_test = apply_scalers(X_test, scalers)

    def get_dl(x: np.ndarray, y_: np.ndarray, shuffle: bool = False):
        return DataLoader(
            TensorDataset(
                torch.tensor(x[:, None, :, :], dtype=torch.float32),
                torch.tensor(y_, dtype=torch.long),
            ),
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )

    train_dl = get_dl(X_train, y_train, shuffle=True)
    val_dl = get_dl(X_val, y_val, shuffle=False)
    test_dl = get_dl(X_test, y_test, shuffle=False)

    crop_samples = int(cfg.crop_seconds * sfreq)
    if crop_samples <= 0:
        raise ValueError("crop_samples computed <= 0. Check crop_seconds and sfreq.")

    # 4) Model + optimizer + scheduler
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

    logger.info("Training started")

    for ep in range(cfg.epochs):
        model.train()
        t_loss = 0.0

        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            # Repeat random crops per batch to increase effective training samples
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

        # Average loss across batches and crop repeats
        avg_loss = t_loss / max(1, (len(train_dl) * cfg.crops_per_epoch))

        # 5) Validation (optional test-time voting over multiple crops)
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)

                if cfg.tta_voting:
                    B, _, _, Tm = xb.shape
                    logits_sum = torch.zeros(B, 4, device=device)
                    stride = max(1, int(cfg.tta_stride * Tm))

                    for start in range(0, max(1, Tm - crop_samples + 1), stride):
                        end = start + crop_samples
                        if end <= Tm:
                            logits_sum += model(xb[:, :, :, start:end])

                    # Always include the final crop
                    logits_sum += model(xb[:, :, :, Tm - crop_samples : Tm])
                    preds = logits_sum.argmax(1)
                else:
                    preds = model(xb[:, :, :, :crop_samples]).argmax(1)

                correct += (preds == yb).sum().item()
                total += int(yb.size(0))

        val_acc = correct / max(1, total)
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(f"Ep {ep+1:03d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {lr_now:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ct = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_ct += 1
            if patience_ct >= cfg.patience:
                logger.info("Early stopping")
                break

    # 6) Test evaluation (load best checkpoint)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            if cfg.tta_voting:
                B, _, _, Tm = xb.shape
                logits_sum = torch.zeros(B, 4, device=device)
                stride = max(1, int(cfg.tta_stride * Tm))

                for start in range(0, max(1, Tm - crop_samples + 1), stride):
                    end = start + crop_samples
                    if end <= Tm:
                        logits_sum += model(xb[:, :, :, start:end])

                logits_sum += model(xb[:, :, :, Tm - crop_samples : Tm])
                preds = logits_sum.argmax(1)
            else:
                preds = model(xb[:, :, :, :crop_samples]).argmax(1)

            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_true.extend(yb.detach().cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    test_acc = float((all_preds == all_true).mean())
    kappa = float(cohen_kappa_score(all_true, all_preds))
    cm = confusion_matrix(all_true, all_preds)
    report = classification_report(all_true, all_preds, target_names=["Left", "Right", "Foot", "Tongue"])

    # Save per-run artifacts
    (run_dir / "confusion_matrix.txt").write_text(str(cm), encoding="utf-8")
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    metrics = {
        "subject": subject_id,
        "seed": int(seed),
        "gdf_path": str(gdf_path),
        "device": str(device),
        "val_best_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
        "timestamp": datetime.datetime.now().isoformat(),
        "cfg": cfg_to_dict(cfg),
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"FINAL | Subject={subject_id} Seed={seed} | Test Acc: {test_acc*100:.2f}% | Kappa: {kappa:.4f}")
    logger.info(f"Artifacts: {run_dir}")

    # Avoid duplicate handlers across repeated runs
    logger.handlers.clear()

    return metrics


# -----------------------------
# Aggregation helpers
# -----------------------------
def mean_std(x):
    x = np.array(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=0))


def write_summary(base_run_dir: Path, all_metrics: list):
    """Write per-run CSV and aggregate JSON summary under the run directory."""
    base_run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_run_dir / "summary.csv"
    fieldnames = ["subject", "seed", "val_best_acc", "test_acc", "kappa", "run_dir"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in all_metrics:
            row = {
                "subject": m["subject"],
                "seed": m["seed"],
                "val_best_acc": m["val_best_acc"],
                "test_acc": m["test_acc"],
                "kappa": m["kappa"],
                "run_dir": m.get("run_dir", ""),
            }
            w.writerow(row)

    subjects = sorted(set(m["subject"] for m in all_metrics))
    seeds = sorted(set(m["seed"] for m in all_metrics))

    by_subject = {s: [] for s in subjects}
    for m in all_metrics:
        by_subject[m["subject"]].append(m["test_acc"])

    subject_means = {s: mean_std(by_subject[s])[0] for s in subjects}
    subject_stds = {s: mean_std(by_subject[s])[1] for s in subjects}

    overall_mean, overall_std = mean_std([m["test_acc"] for m in all_metrics])

    summary = {
        "n_runs": len(all_metrics),
        "subjects": subjects,
        "seeds": seeds,
        "overall_test_acc_mean": overall_mean,
        "overall_test_acc_std": overall_std,
        "per_subject_test_acc_mean": subject_means,
        "per_subject_test_acc_std": subject_stds,
        "note": "Within-subject split per subject file (AxxT.gdf).",
        "created_at": datetime.datetime.now().isoformat(),
    }

    with open(base_run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Runs: {len(all_metrics)} | Subjects: {len(subjects)} | Seeds: {len(seeds)}")
    print(f"Overall Test Acc mean±std: {overall_mean*100:.2f}% ± {overall_std*100:.2f}%")
    print(f"Saved: {csv_path}")
    print(f"Saved: {base_run_dir / 'summary.json'}")


# -----------------------------
# Main
# -----------------------------
def main():
    # Dataset directory can be provided via BCICIV2A_DIR. Falls back to ./data if unset.
    base_dir = Path(os.environ.get("BCICIV2A_DIR", "data/BCICIV_2a_gdf"))

    run_root = Path("runs")

    subjects = [f"A{str(i).zfill(2)}T" for i in range(1, 10)]  # A01T..A09T
    seeds = [42, 43, 44, 45, 46]

    n_runs = len(subjects) * len(seeds)
    print(f"Device: {device}")
    print(f"Data dir: {base_dir}")
    print(f"Planned runs: {len(subjects)} subjects × {len(seeds)} seeds = {n_runs}")
    print(f"Run outputs: {run_root.resolve()}")

    all_metrics = []

    for subj in subjects:
        gdf_path = base_dir / f"{subj}.gdf"

        if not gdf_path.exists():
            print(f"[SKIP] Missing file: {gdf_path}")
            continue

        subject_failed = False

        for seed in seeds:
            try:
                metrics = run_one(
                    subject_id=subj,
                    gdf_path=str(gdf_path),
                    seed=seed,
                    base_run_dir=run_root,
                )
                metrics["run_dir"] = ""  # optional placeholder
                all_metrics.append(metrics)

            except Exception as e:
                print(f"[ERROR] {subj} seed {seed} failed: {e}")
                subject_failed = True
                break

        if subject_failed:
            print(f"[INFO] Skipping remaining seeds for {subj}.")

    write_summary(run_root, all_metrics)


if __name__ == "__main__":
    main()
