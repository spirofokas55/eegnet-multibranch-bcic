"""
Method comparison: EEGNet 7.0 vs Riemannian MDM — BCIC IV-2a

EEGNet values are read from the committed figure (mean ± std over 5 seeds,
within-session holdout split). Riemannian values are loaded from results JSON
(mean ± std over 5 CV folds).

Outputs:
  figures/method_comparison.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ----------------------------------------
# EEGNet results (read from figures/per_subject_accuracy.png)
# mean ± std over 5 seeds, within-session test split
# ----------------------------------------
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

# ----------------------------------------
# Riemannian MDM results (from results/riemannian_results.json)
# mean ± std over 5 CV folds, same subjects
# ----------------------------------------
def load_mdm(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        r["subject"]: (r["mdm_acc_mean"], r["mdm_acc_std"])
        for r in data["results"]
    }


def plot_comparison(eegnet: dict, mdm: dict, out_path: Path):
    subjects = sorted(set(eegnet) & set(mdm))

    eeg_means = np.array([eegnet[s][0] for s in subjects])
    eeg_stds  = np.array([eegnet[s][1] for s in subjects])
    mdm_means = np.array([mdm[s][0]    for s in subjects])
    mdm_stds  = np.array([mdm[s][1]    for s in subjects])

    x = np.arange(len(subjects))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    bars_eeg = ax.bar(x - w/2, eeg_means, w, yerr=eeg_stds,
                      capsize=4, label="EEGNet 7.0\n(mean±std, 5 seeds)",
                      color="#2171b5", error_kw=dict(elinewidth=1.2))
    bars_mdm = ax.bar(x + w/2, mdm_means, w, yerr=mdm_stds,
                      capsize=4, label="Riemannian MDM\n(mean±std, 5-fold CV)",
                      color="#ef6548", error_kw=dict(elinewidth=1.2))

    # Chance line
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Chance (25%)")

    # Delta annotations (MDM - EEGNet)
    for i, s in enumerate(subjects):
        delta = mdm_means[i] - eeg_means[i]
        sign = "+" if delta >= 0 else ""
        color = "#238b45" if delta >= 0 else "#cb181d"
        ax.text(x[i], max(eeg_means[i] + eeg_stds[i], mdm_means[i] + mdm_stds[i]) + 0.025,
                f"{sign}{delta*100:.0f}%", ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("EEGNet 7.0 vs Riemannian MDM — BCIC IV-2a (8 subjects)")
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

    # Overall means in subtitle
    overall_eeg = eeg_means.mean()
    overall_mdm = mdm_means.mean()
    ax.text(0.5, -0.13,
            f"Overall mean — EEGNet: {overall_eeg*100:.1f}%   MDM: {overall_mdm*100:.1f}%   "
            f"(EEGNet note: values read from figure; MDM values from riemannian_results.json)",
            ha="center", transform=ax.transAxes, fontsize=7.5, color="gray")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    # Print table
    print(f"\n{'Subject':<10} {'EEGNet':>10} {'MDM':>10} {'Delta':>8}")
    print("-" * 42)
    for i, s in enumerate(subjects):
        delta = mdm_means[i] - eeg_means[i]
        sign = "+" if delta >= 0 else ""
        print(f"{s:<10} {eeg_means[i]*100:>9.1f}%  {mdm_means[i]*100:>8.1f}%  {sign}{delta*100:.1f}%")
    print("-" * 42)
    print(f"{'Mean':<10} {overall_eeg*100:>9.1f}%  {overall_mdm*100:>8.1f}%  "
          f"{'+' if overall_mdm >= overall_eeg else ''}{(overall_mdm-overall_eeg)*100:.1f}%")


def main():
    json_path = Path("results/riemannian_results.json")
    out_path  = Path("figures/method_comparison.png")

    mdm = load_mdm(json_path)
    plot_comparison(EEGNET, mdm, out_path)


if __name__ == "__main__":
    main()
