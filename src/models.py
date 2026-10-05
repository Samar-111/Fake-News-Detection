import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from src.feature_extraction import get_tfidf_vectorizer
from src.evaluation import evaluate_model_performance
from src.calibration import calibrate_model
from src.transformer_model import TransformerPipelineStub

def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": LinearSVC(C=1.0, random_state=42, dual=False),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric="logloss", random_state=42, n_jobs=-1)
    }

def train_and_benchmark_models(
    df,
    output_model_path="models/calibrated_production_model.joblib",
    output_vectorizer_path="models/tfidf_vectorizer.joblib",
    metrics_path="data/processed/model_benchmark_results.json"
):
    X = np.asarray(df["text"].tolist(), dtype=object)
    y = np.asarray(df["label"].tolist(), dtype=int)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    vectorizer = get_tfidf_vectorizer(max_features=8000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    Path(output_vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_vectorizer_path)
    
    models = get_candidate_models()
    benchmark_results = []
    trained_calibrated_models = {}
    
    for name, base_model in models.items():
        base_model.fit(X_train, y_train)
        
        calibrated = calibrate_model(base_model, X_train, y_train, method="sigmoid", cv=3)
        trained_calibrated_models[name] = calibrated
        
        preds = calibrated.predict(X_test)
        probs = calibrated.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model_performance(y_test, preds, probs, model_name=name)
        benchmark_results.append(metrics)
        
    transformer_metrics = TransformerPipelineStub().get_benchmark_metrics()
    benchmark_results.append(transformer_metrics)
    
    best_candidate_name = "Logistic Regression"
    best_f1 = -1
    for r in benchmark_results:
        if r["model_name"] in trained_calibrated_models and r["f1_score"] > best_f1:
            best_f1 = r["f1_score"]
            best_candidate_name = r["model_name"]
            
    production_model = trained_calibrated_models[best_candidate_name]
    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_model, output_model_path)
    
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
        
    return benchmark_results, production_model, vectorizer, (X_test_raw, y_test)
