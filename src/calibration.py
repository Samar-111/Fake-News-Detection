import numpy as np
import joblib
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

def calibrate_model(base_estimator, X_train, y_train, method="sigmoid", cv=5):
    calibrated = CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated

def compute_calibration_curve(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)
    return {
        "brier_score": round(float(brier), 4),
        "fraction_of_positives": [round(float(p), 4) for p in prob_true],
        "mean_predicted_value": [round(float(p), 4) for p in prob_pred]
    }

def get_confidence_tier(prob_fake):
    distance_from_center = abs(prob_fake - 0.5)
    if distance_from_center >= 0.35:
        return "High"
    elif distance_from_center >= 0.15:
        return "Medium"
    else:
        return "Low"

def format_prediction_result(prob_fake):
    pred_label = "Potentially Fake" if prob_fake >= 0.5 else "Credible News Style"
    confidence = get_confidence_tier(prob_fake)
    return {
        "prediction": pred_label,
        "is_fake": bool(prob_fake >= 0.5),
        "probability_fake": round(float(prob_fake * 100), 1),
        "probability_real": round(float((1.0 - prob_fake) * 100), 1),
        "confidence": confidence
    }
