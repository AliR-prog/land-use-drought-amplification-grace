import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

def main():
    parser = argparse.ArgumentParser(description="Train Random Forest on panel data.")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-out", default="results/rf_model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.panel).sort_values(["Year","BasinID"]).copy()

    features = ["fUrban", "fCropland", "fVegetation", "P_anom", "T_anom"]
    target = "TWSAmin_cm"

    # simple time split for quick reproducibility
    years = sorted(df["Year"].unique())
    split_year = years[max(1, len(years)//2)-1]
    train = df[df["Year"] <= split_year]
    test = df[df["Year"] > split_year]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test) if len(test) > 0 else []

    rows = []
    rows.append({
        "Metric":"Train",
        "R2": r2_score(y_train, pred_train),
        "RMSE": mean_squared_error(y_train, pred_train, squared=False),
        "MAE": mean_absolute_error(y_train, pred_train)
    })
    if len(test) > 0:
        rows.append({
            "Metric":"Test",
            "R2": r2_score(y_test, pred_test),
            "RMSE": mean_squared_error(y_test, pred_test, squared=False),
            "MAE": mean_absolute_error(y_test, pred_test)
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"Saved metrics to {args.output}")
    print(f"Saved model to {args.model_out}")

if __name__ == "__main__":
    main()
