import argparse
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Compute SHAP values and counterfactual amplification.")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--model", default="results/rf_model.joblib")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.panel).copy()
    model = joblib.load(args.model)

    features = ["fUrban", "fCropland", "fVegetation", "P_anom", "T_anom"]

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df[features])

    shap_mean = pd.DataFrame({
        "Feature": features,
        "MeanAbsSHAP": np.abs(shap_values).mean(axis=0)
    })
    shap_mean.to_csv(f"{args.output_prefix}_shap_importance.csv", index=False)

    plt.figure(figsize=(7,4))
    shap.summary_plot(shap_values, df[features], show=False)
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Counterfactual: hold land use at first available year in each basin
    baseline = df.sort_values("Year").groupby("BasinID")[["fUrban","fCropland","fVegetation"]].first().reset_index()
    base_map = baseline.set_index("BasinID").to_dict(orient="index")

    X_obs = df[features].copy()
    y_obs = model.predict(X_obs)

    X_cf = X_obs.copy()
    for i, row in df.iterrows():
        b = row["BasinID"]
        X_cf.loc[i, "fUrban"] = base_map[b]["fUrban"]
        X_cf.loc[i, "fCropland"] = base_map[b]["fCropland"]
        X_cf.loc[i, "fVegetation"] = base_map[b]["fVegetation"]

    y_cf = model.predict(X_cf)
    amp = y_obs - y_cf

    out = df[["BasinID","Year"]].copy()
    out["Pred_obs"] = y_obs
    out["Pred_cf"] = y_cf
    out["Amplification_cm"] = amp
    out.to_csv(f"{args.output_prefix}_counterfactual.csv", index=False)

    basin_summary = out.groupby("BasinID")["Amplification_cm"].agg(["mean","median","std","min","max"]).reset_index()
    basin_summary.to_csv(f"{args.output_prefix}_counterfactual_summary.csv", index=False)
    print("Saved SHAP and counterfactual outputs.")

if __name__ == "__main__":
    main()
