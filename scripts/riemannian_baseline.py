"""
Riemannian Geometry Baseline — BCIC IV-2a (4-class motor imagery)

Pipeline:
  Raw EEG -> bandpass -> epoch -> covariance matrices
           -> MDM (Minimum Distance to Mean, Riemannian metric)
           -> 5-fold stratified CV accuracy

Also runs a Tangent Space + LDA pipeline for comparison.

Outputs:
  runs/riemannian_results.json
  runs/riemannian_summary.csv
"""

import os
import json
import csv
import datetime
from pathlib import Path

import numpy as np
import mne
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace


# ----------------------------------------
# Config
# ----------------------------------------
TMIN = 0.5       # seconds post-cue (avoid motor preparation noise)
TMAX = 2.5       # seconds post-cue
FREQ_L = 8.0     # mu/beta band
FREQ_H = 30.0
REJECT_THRESHOLD = 150e-6
N_CHANNELS = 22
N_SPLITS = 5
RANDOM_STATE = 42


# ----------------------------------------
# Data loading (adapted from eegnet7_multisubject.py)
# ----------------------------------------
def load_data(path: str):
    """Returns X (n_epochs, n_channels, n_times) and y (n_epochs,)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_gdf(path, preload=True)
    raw.pick("eeg")

    try:
        raw.set_eeg_reference("average", projection=False)
    except Exception:
        pass

    raw.filter(FREQ_L, FREQ_H, fir_design="firwin")

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
            f"Unexpected event codes.\nPresent: {sorted(present)}\n"
            f"Annotation map: {list(event_id_from_ann.items())[:10]}"
        )

    rel_events = events[np.isin(events[:, 2], list(desired.values()))]

    epochs = mne.Epochs(
        raw,
        rel_events,
        event_id=desired,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        picks="eeg",
        preload=True,
        reject={"eeg": REJECT_THRESHOLD},
    )

    X = epochs.get_data(copy=True)[:, :N_CHANNELS, :]
    inv = {desired["left"]: 0, desired["right"]: 1, desired["foot"]: 2, desired["tongue"]: 3}
    y = np.array([inv[c] for c in epochs.events[:, -1]])

    return X, y


# ----------------------------------------
# Pipelines
# ----------------------------------------
def make_mdm_pipeline():
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("mdm", MDM(metric="riemann")),
    ])


def make_ts_lda_pipeline():
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("lda", LinearDiscriminantAnalysis()),
    ])


# ----------------------------------------
# Single subject
# ----------------------------------------
def run_subject(subject_id: str, gdf_path: str) -> dict:
    print(f"\n[{subject_id}] Loading {gdf_path} ...", flush=True)
    X, y = load_data(gdf_path)
    print(f"[{subject_id}] X={X.shape} y={y.shape} classes={np.unique(y, return_counts=True)}", flush=True)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    mdm_scores = cross_val_score(make_mdm_pipeline(), X, y, cv=cv, scoring="accuracy", n_jobs=1)
    ts_scores  = cross_val_score(make_ts_lda_pipeline(), X, y, cv=cv, scoring="accuracy", n_jobs=1)

    result = {
        "subject": subject_id,
        "n_epochs": int(X.shape[0]),
        "mdm_acc_mean": float(mdm_scores.mean()),
        "mdm_acc_std":  float(mdm_scores.std()),
        "ts_lda_acc_mean": float(ts_scores.mean()),
        "ts_lda_acc_std":  float(ts_scores.std()),
        "mdm_folds": mdm_scores.tolist(),
        "ts_lda_folds": ts_scores.tolist(),
    }

    print(
        f"[{subject_id}] MDM:    {result['mdm_acc_mean']*100:.1f}% ± {result['mdm_acc_std']*100:.1f}%  |  "
        f"TS+LDA: {result['ts_lda_acc_mean']*100:.1f}% ± {result['ts_lda_acc_std']*100:.1f}%",
        flush=True,
    )
    return result


# ----------------------------------------
# Main
# ----------------------------------------
def main():
    base_dir = Path(os.environ.get("BCICIV2A_DIR", r"C:\Users\spiro\BCI-NeuroStart\data\BCICIV_2a_gdf"))
    run_root = Path("runs")
    run_root.mkdir(parents=True, exist_ok=True)

    subjects = [f"A{str(i).zfill(2)}T" for i in range(1, 10)]

    print(f"Data dir : {base_dir}")
    print(f"Subjects : {subjects}")
    print(f"CV folds : {N_SPLITS}")
    print(f"Band     : {FREQ_L}-{FREQ_H} Hz  |  Epoch: {TMIN}-{TMAX}s")

    all_results = []

    for subj in subjects:
        gdf_path = base_dir / f"{subj}.gdf"
        if not gdf_path.exists():
            print(f"[SKIP] {gdf_path} not found")
            continue
        try:
            result = run_subject(subj, str(gdf_path))
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] {subj}: {e}")

    if not all_results:
        print("No results — check BCICIV2A_DIR path.")
        return

    # Save JSON
    output = {
        "created_at": datetime.datetime.now().isoformat(),
        "config": {
            "tmin": TMIN, "tmax": TMAX,
            "freq_l": FREQ_L, "freq_h": FREQ_H,
            "n_channels": N_CHANNELS,
            "cv_folds": N_SPLITS,
            "covariance_estimator": "oas",
        },
        "results": all_results,
        "summary": {
            "mdm_overall_mean": float(np.mean([r["mdm_acc_mean"] for r in all_results])),
            "ts_lda_overall_mean": float(np.mean([r["ts_lda_acc_mean"] for r in all_results])),
        },
    }

    json_path = run_root / "riemannian_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Save CSV
    csv_path = run_root / "riemannian_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "n_epochs", "mdm_acc_mean", "mdm_acc_std",
                                           "ts_lda_acc_mean", "ts_lda_acc_std"])
        w.writeheader()
        for r in all_results:
            w.writerow({k: r[k] for k in w.fieldnames})

    print("\n=== SUMMARY ===")
    print(f"MDM overall:    {output['summary']['mdm_overall_mean']*100:.1f}%")
    print(f"TS+LDA overall: {output['summary']['ts_lda_overall_mean']*100:.1f}%")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
