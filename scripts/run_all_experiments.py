import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
import pandas as pd
from src.data_pipeline import prepare_unified_dataset
from src.models import train_and_benchmark_models
from src.error_analysis import run_systematic_error_analysis
from src.leakage_experiments import run_data_leakage_experiments

def main():
    df, summary = prepare_unified_dataset()
    print("Dataset ready:", df.shape)

    benchmarks, prod_model, vectorizer, (X_test_raw, y_test) = train_and_benchmark_models(df)
    print("Trained and benchmarked models successfully.")

    X_test_vec = vectorizer.transform(X_test_raw)
    preds = prod_model.predict(X_test_vec)
    probs = prod_model.predict_proba(X_test_vec)[:, 1]

    df_test_sample = df.sample(n=len(y_test), random_state=42)
    error_rep = run_systematic_error_analysis(df_test_sample, y_test, preds, probs)
    print("Error analysis completed.")

    leak_rep = run_data_leakage_experiments(df)
    print("Leakage experiments completed.")

    print("\n--- BENCHMARK RESULTS ---")
    for b in benchmarks:
        print(f"{b['model_name']:<25} | Acc: {b['accuracy']:.4f} | Prec: {b['precision']:.4f} | Rec: {b['recall']:.4f} | F1: {b['f1_score']:.4f}")

if __name__ == "__main__":
    main()
