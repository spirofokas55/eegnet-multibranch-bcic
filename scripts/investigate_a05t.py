"""
Diagnostic: why does MDM fail on A05T while EEGNet gets 72%?

Checks per subject:
  - Epoch count and class balance
  - Per-channel signal variance (proxy for noise/artifact level)
  - Covariance matrix condition number (ill-conditioned = bad for Riemannian)
  - Log-Euclidean distance between class-mean covariances (class separability)

Compares A05T against A01T (strong subject) and A09T (MDM beats EEGNet).
"""

import os
from pathlib import Path

import numpy as np
import mne
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance

TMIN = 0.5
TMAX = 2.5
FREQ_L = 8.0
FREQ_H = 30.0
REJECT_THRESHOLD = 150e-6
N_CHANNELS = 22

SUBJECTS = ["A01T", "A05T", "A09T"]


def load_data(path: str):
    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_gdf(path, preload=True)
    raw.pick("eeg")
    try:
        raw.set_eeg_reference("average", projection=False)
    except Exception:
        pass
    raw.filter(FREQ_L, FREQ_H, fir_design="firwin")

    events, _ = mne.events_from_annotations(raw)
    present = set(events[:, 2].tolist())

    desired_769 = {"left": 769, "right": 770, "foot": 771, "tongue": 772}
    desired_7   = {"left": 7,   "right": 8,   "foot": 9,   "tongue": 10}
    if all(v in present for v in desired_769.values()):
        desired = desired_769
    else:
        desired = desired_7

    rel_events = events[np.isin(events[:, 2], list(desired.values()))]
    epochs = mne.Epochs(raw, rel_events, event_id=desired,
                        tmin=TMIN, tmax=TMAX, baseline=None,
                        picks="eeg", preload=True,
                        reject={"eeg": REJECT_THRESHOLD})

    X = epochs.get_data(copy=True)[:, :N_CHANNELS, :]
    inv = {desired["left"]: 0, desired["right"]: 1,
           desired["foot"]: 2, desired["tongue"]: 3}
    y = np.array([inv[c] for c in epochs.events[:, -1]])
    return X, y


def analyse(subject_id: str, X: np.ndarray, y: np.ndarray):
    print(f"\n{'='*50}")
    print(f"  {subject_id}")
    print(f"{'='*50}")

    # Class balance
    uniq, cnt = np.unique(y, return_counts=True)
    labels = ["Left", "Right", "Foot", "Tongue"]
    print(f"Epochs: {len(y)}  |  Class counts: { {labels[u]: c for u,c in zip(uniq,cnt)} }")

    # Signal variance per channel (mean across epochs)
    var_per_ch = X.var(axis=2).mean(axis=0)  # (C,)
    print(f"Signal variance — mean: {var_per_ch.mean():.2e}  "
          f"min: {var_per_ch.min():.2e}  max: {var_per_ch.max():.2e}  "
          f"CV: {var_per_ch.std()/var_per_ch.mean():.2f}")

    # Covariance matrices
    cov_est = Covariances(estimator="oas")
    covs = cov_est.fit_transform(X)          # (n_epochs, C, C)

    # Condition numbers
    conds = np.linalg.cond(covs)
    print(f"Cov condition number — median: {np.median(conds):.1f}  "
          f"p95: {np.percentile(conds, 95):.1f}  max: {conds.max():.1f}")

    # Class-mean covariances and inter-class distance
    class_means = []
    for c in range(4):
        idx = y == c
        if idx.sum() < 2:
            continue
        cm = mean_covariance(covs[idx], metric="riemann")
        class_means.append(cm)

    # Pairwise log-Euclidean distances between class means
    from pyriemann.utils.distance import distance
    dists = []
    for i in range(len(class_means)):
        for j in range(i+1, len(class_means)):
            d = distance(class_means[i], class_means[j], metric="riemann")
            dists.append(d)
    print(f"Inter-class Riemannian distances — "
          f"mean: {np.mean(dists):.4f}  min: {np.min(dists):.4f}  max: {np.max(dists):.4f}")


def main():
    base_dir = Path(os.environ.get("BCICIV2A_DIR",
                    r"C:\Users\spiro\BCI-NeuroStart\data\BCICIV_2a_gdf"))

    for subj in SUBJECTS:
        gdf_path = base_dir / f"{subj}.gdf"
        if not gdf_path.exists():
            print(f"[SKIP] {gdf_path} not found")
            continue
        X, y = load_data(str(gdf_path))
        analyse(subj, X, y)

    print("\nKey: low inter-class distance = poor Riemannian separability")


if __name__ == "__main__":
    main()
