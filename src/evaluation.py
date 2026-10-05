import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss
)

def evaluate_model_performance(y_true, y_pred, y_prob=None, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = None
    brier = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
        except Exception:
            auc = None
            brier = None

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc is not None else None,
        "brier_score": round(float(brier), 4) if brier is not None else None,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "reasoning": {
            "false_positive_impact": "Classifying genuine journalism as fake creates censorship risks and erodes reader trust in legitimate institutions.",
            "false_negative_impact": "Classifying viral misinformation as authentic allows misleading falsehoods to spread unchecked without verification tags.",
            "recommendation": "In production, high precision minimizes unfair censorship, while high recall catches widespread viral falsehoods. An F1-score with probability thresholds provides balanced control."
        }
    }
