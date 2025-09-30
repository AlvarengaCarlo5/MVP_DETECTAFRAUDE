import json, requests, pandas as pd

CSV = "creditcard.csv"  # dataset local
df = pd.read_csv(CSV).head(5)

FEATURES = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
payload = {
    "data": df[FEATURES].to_dict(orient="records"),
    "threshold": 0.40   # opcional; se omitir, usa o salvo no config.json
}

resp = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=30)
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2)[:1500], "...")