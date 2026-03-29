import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Build basin-year panel dataset.")
    parser.add_argument("--grace", required=True)
    parser.add_argument("--climate", required=True)
    parser.add_argument("--lulc", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    grace = pd.read_csv(args.grace)
    climate = pd.read_csv(args.climate)
    lulc = pd.read_csv(args.lulc)

    panel = grace.merge(climate, on=["BasinID", "Year"], how="inner")
    panel = panel.merge(lulc, on=["BasinID", "Year"], how="inner")

    # Keep only the predictors used in the manuscript
    cols = ["BasinID", "Year", "TWSAmin_cm", "P_anom", "T_anom", "fUrban", "fCropland", "fVegetation"]
    panel = panel[cols].copy()
    panel.to_csv(args.output, index=False)
    print(f"Saved panel to {args.output}")

if __name__ == "__main__":
    main()
