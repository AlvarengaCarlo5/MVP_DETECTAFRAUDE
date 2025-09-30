from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
import pandas as pd
import numpy as np
from joblib import load

app = FastAPI(title="Fraud-API")

MODEL = load("models/fraud_model.joblib")
CFG = json.load(open("models/config.json", encoding="utf-8"))
FEATURES = CFG["feature_order"]
THRESHOLD = float(CFG["threshold"])

class Record(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float; V9: float
    V10: float; V11: float; V12: float; V13: float; V14: float; V15: float; V16: float; V17: float; V18: float; V19: float
    V20: float; V21: float; V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float

class BatchRequest(BaseModel):
    data: List[Record]
    threshold: float | None = None

@app.get("/health")
def health():
    return {"status":"ok","threshold":THRESHOLD}

@app.post("/predict")
def predict(req: BatchRequest):
    th = float(req.threshold) if req.threshold is not None else THRESHOLD
    rows = [r.dict() for r in req.data]
    df = pd.DataFrame(rows)
    miss = [c for c in FEATURES if c not in df.columns]
    if miss: return {"error": f"faltam colunas: {miss}"}
    X = df[FEATURES].values
    if hasattr(MODEL,"predict_proba"):
        prob = MODEL.predict_proba(X)[:,1]
    else:
        s = MODEL.decision_function(X); prob = 1/(1+np.exp(-s))
    pred = (prob >= th).astype(int)
    df_out = df.copy()
    df_out["fraud_prob"] = prob
    df_out["fraud_pred"] = pred
    return {"threshold": th, "results": df_out.to_dict(orient="r