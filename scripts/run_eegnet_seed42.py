"""Run EEGNet sweep with a single seed (42) to produce exact per-subject numbers."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import eegnet7_multisubject as m

base_dir = Path(r"C:\Users\spiro\BCI-NeuroStart\data\BCICIV_2a_gdf")
run_root = Path("runs")
subjects = [f"A{str(i).zfill(2)}T" for i in range(1, 10)]

all_metrics = []

for subj in subjects:
    gdf_path = base_dir / f"{subj}.gdf"
    if not gdf_path.exists():
        print(f"[SKIP] {gdf_path}")
        continue
    try:
        metrics = m.run_one(subj, str(gdf_path), seed=42, base_run_dir=run_root)
        all_metrics.append(metrics)
    except Exception as e:
        print(f"[ERROR] {subj}: {e}")

m.write_summary(run_root, all_metrics)

out = run_root / "eegnet_seed42.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2)

print(f"\nSaved: {out}")
print("\nPer-subject test accuracy:")
for r in all_metrics:
    print(f"  {r['subject']}: {r['test_acc']*100:.1f}%")
