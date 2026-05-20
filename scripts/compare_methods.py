"""
Method comparison: EEGNet 7.0 vs Riemannian MDM vs TS+PCA+LDA — BCIC IV-2a

EEGNet values: mean ± std over 5 seeds, read from figures/per_subject_accuracy.png
               (original 5-seed run; JSON not preserved).
Riemannian values: loaded from results/riemannian_results.json
                   (mean ± std over 5-fold CV).

Outputs:
  figures/method_comparison.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# EEGNet 7.0 — mean ± std over 5 seeds, within-session holdout split.
# Values read from figures/per_subject_accuracy.png (original run results).
EEGNET = {
    "A01T": (0.78, 0.05),
    "A02T": (0.52, 0.04),
    "A03T": (0.83, 0.02),
    "A05T": (0.72, 0.03),
    "A06T": (0.49, 0.05),
    "A07T": (0.85, 0.04),
    "A08T": (0.81, 0.05),
    "A09T": (0.65, 0.07),
}


def load_riemannian(json_path: Path) -> tuple[dict, dict]:
    """Returns (mdm, ts_pca_lda) dicts of {subject: (mean, std)}."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    mdm, ts = {}, {}
    for r in data["results"]:
        s = r["subject"]
        mdm[s] = (r["mdm_acc_mean"], r["mdm_acc_std"])
        ts[s]  = (r["ts_pca_lda_acc_mean"], r["ts_pca_lda_acc_std"])
    return mdm, ts


def plot_comparison(eegnet: dict, mdm: dict, ts: dict, out_path: Path):
    subjects = sorted(set(eegnet) & set(mdm) & set(ts))

    eeg_means = np.array([eegnet[s][0] for s in subjects])
    mdm_means = np.array([mdm[s][0]    for s in subjects])
    mdm_stds  = np.array([mdm[s][1]    for s in subjects])
    ts_means  = np.array([ts[s][0]     for s in subjects])
    ts_stds   = np.array([ts[s][1]     for s in subjects])

    x = np.arange(len(subjects))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.bar(x - w,   eeg_means, w, label="EEGNet 7.0\n(seed=42, test split)",
           color="#2171b5")
    ax.bar(x,       mdm_means, w, yerr=mdm_stds, capsize=3,
           label="Riemannian MDM\n(mean±std, 5-fold CV)", color="#ef6548",
           error_kw=dict(elinewidth=1.2))
    ax.bar(x + w,   ts_means,  w, yerr=ts_stds,  capsize=3,
           label="TS+PCA+LDA\n(mean±std, 5-fold CV)", color="#41ab5d",
           error_kw=dict(elinewidth=1.2))

    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Chance (25%)")

    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("EEGNet 7.0 vs Riemannian MDM vs TS+PCA+LDA — BCIC IV-2a (9 subjects)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

    overall_eeg = eeg_means.mean()
    overall_mdm = mdm_means.mean()
    overall_ts  = ts_means.mean()
    ax.text(0.5, -0.13,
            f"Overall mean — EEGNet: {overall_eeg*100:.1f}%  "
            f"MDM: {overall_mdm*100:.1f}%  "
            f"TS+PCA+LDA: {overall_ts*100:.1f}%",
            ha="center", transform=ax.transAxes, fontsize=8, color="gray")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    # Print table
    print(f"\n{'Subject':<10} {'EEGNet':>9} {'MDM':>9} {'TS+PCA+LDA':>12}")
    print("-" * 45)
    for i, s in enumerate(subjects):
        print(f"{s:<10} {eeg_means[i]*100:>8.1f}%  {mdm_means[i]*100:>8.1f}%  {ts_means[i]*100:>11.1f}%")
    print("-" * 45)
    print(f"{'Mean':<10} {overall_eeg*100:>8.1f}%  {overall_mdm*100:>8.1f}%  {overall_ts*100:>11.1f}%")


def main():
    riem_json = Path("results/riemannian_results.json")
    out_path  = Path("figures/method_comparison.png")

    mdm, ts = load_riemannian(riem_json)
    # Only plot subjects present in both EEGNet and Riemannian results
    eegnet = {s: v for s, v in EEGNET.items() if s in mdm}
    plot_comparison(eegnet, mdm, ts, out_path)


if __name__ == "__main__":
    main()
