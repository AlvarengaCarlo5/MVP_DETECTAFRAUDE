import argparse, json
import pandas as pd
from joblib import load
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV sem coluna Class (Time,V1..V28,Amount)")
    ap.add_argument("--model", default="models/fraud_model.joblib")
    ap.add_argument("--config", default="models/config.json")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--threshold", type=float, default=None, help="sobrepõe config.json")
    args = ap.parse_args()

    model = load(args.model)
    cfg = json.load(open(args.config, encoding="utf-8"))
    feats = cfg["feature_order"]
    thr = float(args.threshold) if args.threshold is not None else float(cfg["threshold"])

    df = pd.read_csv(args.input)
    miss = [c for c in feats if c not in df.columns]
    if miss: raise ValueError(f"faltam colunas: {miss}")

    X = df[feats].values
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:,1]
    else:
        s = model.decision_function(X); prob = 1/(1+np.exp(-s))
    pred = (prob >= thr).astype(int)

    out = df.copy()
    out["fraud_prob"] = prob
    out["fraud_pred"] = pred
    out.to_csv(args.output, index=False)
    print(f"OK: salvo {args.output} (thr={thr})")

if __name__ == "__main__":
    main()