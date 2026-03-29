import os
import subprocess
from pathlib import Path

base = Path(__file__).resolve().parents[1]
example = base / "example_data"
results = base / "results"
results.mkdir(exist_ok=True)

subprocess.run([
    "python", str(base / "src" / "01_build_panel.py"),
    "--grace", str(example / "grace_basin_year.csv"),
    "--climate", str(example / "climate_basin_year.csv"),
    "--lulc", str(example / "lulc_basin_year.csv"),
    "--output", str(results / "panel.csv"),
], check=True)

subprocess.run([
    "python", str(base / "src" / "02_train_rf.py"),
    "--panel", str(results / "panel.csv"),
    "--output", str(results / "model_metrics.csv"),
    "--model-out", str(results / "rf_model.joblib"),
], check=True)

subprocess.run([
    "python", str(base / "src" / "03_shap_counterfactual.py"),
    "--panel", str(results / "panel.csv"),
    "--model", str(results / "rf_model.joblib"),
    "--output-prefix", str(results / "final"),
], check=True)

print("Quick test completed. Check the results/ folder.")
