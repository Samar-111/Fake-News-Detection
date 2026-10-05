import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.calibration import format_prediction_result
from src.explainability import ModelExplainer

app = FastAPI(
    title="Fake News Intelligence API",
    version="1.0.0",
    description="Real-time calibrated misinformation risk classification and feature explainability service."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = Path("models/calibrated_production_model.joblib")
vec_path = Path("models/tfidf_vectorizer.joblib")
metrics_path = Path("data/processed/model_benchmark_results.json")
leakage_path = Path("data/processed/leakage_comparison.json")
errors_path = Path("data/processed/error_analysis.json")
static_dist = Path("web/dist")

model = None
vectorizer = None
explainer = None

def get_resources():
    global model, vectorizer, explainer
    if model is None or vectorizer is None:
        if not model_path.exists() or not vec_path.exists():
            raise RuntimeError("Model artifacts not found. Run scripts/run_all_experiments.py first.")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        explainer = ModelExplainer(vectorizer, model)
    return model, vectorizer, explainer

class NewsRequest(BaseModel):
    text: str = Field(..., min_length=10, description="The article body or headline to evaluate.")

@app.get("/health")
def health():
    m, v, _ = get_resources()
    return {
        "status": "healthy",
        "model_loaded": True,
        "vocabulary_size": len(v.vocabulary_),
        "classifier_type": type(m).__name__
    }

@app.post("/predict")
def predict_news(payload: NewsRequest):
    m, v, exp = get_resources()
    text = payload.text.strip()
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Article text must contain at least 10 characters.")

    vec = v.transform([text])
    prob_fake = float(m.predict_proba(vec)[0][1])
    result = format_prediction_result(prob_fake)
    attribution = exp.explain_fast_attribution(text, top_k=5)

    return {
        "prediction": result["prediction"],
        "is_fake": result["is_fake"],
        "probability_fake": result["probability_fake"],
        "probability_real": result["probability_real"],
        "confidence": result["confidence"],
        "key_indicators": attribution.get("key_indicators", []),
        "disclaimer": "This system operates as a stylometric misinformation-risk classifier. It analyzes linguistic patterns and source attributions, and does not replace human investigative journalism or factual ground-truth verification."
    }

@app.post("/explain")
def explain_news(payload: NewsRequest):
    m, v, exp = get_resources()
    text = payload.text.strip()
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Article text must contain at least 10 characters.")

    vec = v.transform([text])
    prob_fake = float(m.predict_proba(vec)[0][1])
    result = format_prediction_result(prob_fake)
    attribution = exp.explain_fast_attribution(text, top_k=8)

    return {
        "prediction": result["prediction"],
        "is_fake": result["is_fake"],
        "probability_fake": result["probability_fake"],
        "probability_real": result["probability_real"],
        "confidence": result["confidence"],
        "fake_signals": [{"word": w, "weight": s} for w, s in attribution["fake_signals"]],
        "real_signals": [{"word": w, "weight": s} for w, s in attribution["real_signals"]],
        "key_indicators": attribution.get("key_indicators", []),
        "disclaimer": "This system evaluates writing style and lexical indicators. It flags stylistic misinformation risks rather than verifying factual veracity."
    }

@app.get("/metrics")
def get_benchmarks():
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.get("/leakage")
def get_leakage():
    if leakage_path.exists():
        with open(leakage_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.get("/errors")
def get_errors():
    if errors_path.exists():
        with open(errors_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

if static_dist.exists():
    app.mount("/", StaticFiles(directory="web/dist", html=True), name="static")
