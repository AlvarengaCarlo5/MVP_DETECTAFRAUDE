import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    confusion_matrix, precision_recall_fscore_support,
    precision_recall_curve, roc_curve, classification_report
)
import matplotlib.pyplot as plt

SEED = 42
FEATURES = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

def load_csv(path):
    df = pd.read_csv(path)
    miss = [c for c in ["Class"] + FEATURES if c not in df.columns]
    if miss:
        raise ValueError(f"Faltam colunas: {miss}")
    return df

def split_data(df, split_type="random", test_size=0.20, val_size=0.0):
    """
    random: separa TESTE primeiro; opcionalmente separa VALIDACAO do restante.
    temporal: ordena por Time e corta no fim (hold-out temporal).
    """
    if split_type == "temporal":
        df = df.sort_values("Time").reset_index(drop=True)
        n = len(df)
        n_test = int(test_size * n)
        trainval = df.iloc[: n - n_test]
        test = df.iloc[n - n_test :]
        if val_size > 0:
            n_val = int(val_size * n)
            train = trainval.iloc[: len(trainval) - n_val]
            val   = trainval.iloc[len(trainval) - n_val :]
        else:
            train, val = trainval, None
    else:
        X, y = df[FEATURES], df["Class"].astype(int)
        X_trval, X_te, y_trval, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=SEED
        )
        if val_size > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_trval, y_trval,
                test_size=val_size/(1.0 - test_size),
                stratify=y_trval, random_state=SEED
            )
            train = pd.concat([X_tr, y_tr], axis=1)
            val   = pd.concat([X_val, y_val], axis=1)
        else:
            train = pd.concat([X_trval, y_trval], axis=1)
            val   = None
        test  = pd.concat([X_te, y_te], axis=1)
    return train, val, test

def build_pipeline():
    preproc = ColumnTransformer(
        [("scale", StandardScaler(), ["Time", "Amount"])],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1
    )
    return Pipeline([("prep", preproc), ("clf", rf)])

def evaluate(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    ap  = float(average_precision_score(y_true, y_prob))
    roc = float(roc_auc_score(y_true, y_prob))
    f1  = float(f1_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "auc_pr": ap, "roc_auc": roc, "f1_at_thr": f1, "threshold": thr,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)
    }

def save_plots(y_true, y_prob, thr, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= thr).astype(int)

    # 1) PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(rec, prec, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pr_curve.png", dpi=120)
    plt.close()

    # 2) ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"ROC AUC={roc:.4f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=120)
    plt.close()

    # 3) Confusion Matrix (no threshold escolhido)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="viridis")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="w")
    plt.xticks([0,1], ["Legítimo","Fraude"])
    plt.yticks([0,1], ["Legítimo","Fraude"])
    plt.xlabel("Predito"); plt.ylabel("Real")
    plt.title(f"Confusion Matrix (thr={thr})")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=120)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="creditcard.csv")
    ap.add_argument("--split", choices=["random","temporal"], default="random")
    ap.add_argument("--test_size", type=float, default=0.20,
                    help="proporção do TESTE (ex.: 0.20 = 80/20)")
    ap.add_argument("--val_size", type=float, default=0.0,
                    help="proporção da VALIDAÇÃO (0.0 = sem validação separada)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="se informado, usa este threshold; se None e val_size>0, escolhe por F1 na validação; senão 0.5")
    args = ap.parse_args()

    df = load_csv(args.data)
    train, val, test = split_data(
        df, split_type=args.split, test_size=args.test_size, val_size=args.val_size
    )

    X_tr, y_tr = train[FEATURES], train["Class"].astype(int)
    X_te, y_te = test[FEATURES],  test["Class"].astype(int)

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    # Threshold
    if args.threshold is not None:
        best_thr = float(args.threshold)
    elif args.val_size > 0 and val is not None:
        X_val, y_val = val[FEATURES], val["Class"].astype(int)
        y_val_prob = pipe.predict_proba(X_val)[:,1]
        best_thr, best_f1 = 0.5, -1.0
        for t in [i/100 for i in range(10,91,5)]:
            yv = (y_val_prob >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_val, yv, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1, best_thr = float(f1), float(t)
    else:
        best_thr = 0.5

    # Avaliação no TESTE
    y_te_prob = pipe.predict_proba(X_te)[:,1]
    metrics = evaluate(y_te, y_te_prob, thr=best_thr)

    # Salvar modelo e config
    Path("models").mkdir(exist_ok=True)
    dump(pipe, "models/fraud_model.joblib")
    cfg = {
        "feature_order": FEATURES,
        "threshold": best_thr,
        "split_type": args.split,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "metrics_test": metrics,
    }
    with open("models/config.json","w",encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Salvar relatórios + gráficos
    rep = Path("reports"); rep.mkdir(exist_ok=True)
    with open(rep / "classification_report.txt","w",encoding="utf-8") as f:
        f.write(classification_report(y_te, (y_te_prob>=best_thr).astype(int), digits=4))
    with open(rep / "metrics.json","w",encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    save_plots(y_te, y_te_prob, best_thr, rep)

    print("\n=== Treino concluído ===")
    print("Split:", args.split)
    print("Threshold escolhido (val/test):", best_thr)
    print("Métricas no TESTE:", json.dumps(metrics, indent=2))
    print("Gráficos salvos em:", str(rep.resolve()))

if __name__ == "__main__":
    main()
