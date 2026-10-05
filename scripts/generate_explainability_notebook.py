import nbformat as nbf
from pathlib import Path

def create_explainability_notebook(output_path="notebooks/03_explainability_and_error_analysis.ipynb"):
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell(
"""# 🔍 Model Explainability, Calibration & Systematic Error Analysis

This notebook explores:
1. **Model Explainability (LIME & Token Attributions):** Extracting interpretable positive and negative feature signals.
2. **Probability Calibration:** Evaluating posterior probability reliability via Calibration Curves and Brier scores.
3. **❌ Systematic Error Analysis:** Diagnosing failure modes across satire, clickbait, short texts, and unseen publishers.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

from src.explainability import ModelExplainer
from src.calibration import format_prediction_result, compute_calibration_curve
from src.error_analysis import run_systematic_error_analysis

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## 1. Load Calibrated Production Model and Vectorizer"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""model_path = "../models/calibrated_production_model.joblib"
if not os.path.exists(model_path):
    model_path = "models/calibrated_production_model.joblib"

vec_path = "../models/tfidf_vectorizer.joblib"
if not os.path.exists(vec_path):
    vec_path = "models/tfidf_vectorizer.joblib"

model = joblib.load(model_path)
vectorizer = joblib.load(vec_path)
explainer = ModelExplainer(vectorizer, model)

print("Production pipeline successfully loaded.")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## 2. Explainability in Action: Deconstructing Predictions with Signal Weights"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""fake_sample = "BREAKING NEWS: Shocking revelation shows military officials confirmed secret underground operation in panic!"
real_sample = "WASHINGTON (Reuters) - U.S. government report and federal officials confirmed new economic data according to treasury statements."

for title, sample_text in [("Fake Article Demonstration", fake_sample), ("Real Article Demonstration", real_sample)]:
    vec = vectorizer.transform([sample_text])
    prob_fake = model.predict_proba(vec)[0][1]
    res = format_prediction_result(prob_fake)
    exp = explainer.explain_fast_attribution(sample_text)
    
    badge = "🔴 FAKE" if res['is_fake'] else "🟢 REAL"
    pct = res['probability_fake'] if res['is_fake'] else res['probability_real']
    print(f"\\n{'='*60}")
    print(f"{badge} — {pct:.1f}% | Confidence: {res['confidence']}")
    print(f"Text Snippet: '{sample_text}'")
    print("Important Signals:")
    signals = exp['fake_signals'] if res['is_fake'] else exp['real_signals']
    for word, weight in signals:
        sign = "+" if weight > 0 else ""
        print(f"  {word:<22} {sign}{weight:.2f}")
    if exp.get('key_indicators'):
        print("Key Indicators:", ", ".join(exp['key_indicators']))
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## 3. Visualizing Signal Attributions"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

fake_exp = explainer.explain_fast_attribution(fake_sample)
fake_df = pd.DataFrame(fake_exp['fake_signals'], columns=['Token', 'Weight']).sort_values('Weight', ascending=True)
axes[0].barh(fake_df['Token'], fake_df['Weight'], color='#f43f5e')
axes[0].set_title("🔴 Signals Pushing Towards 'Fake' (Misinformation Risk)", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Attribution Weight")

real_exp = explainer.explain_fast_attribution(real_sample)
real_df = pd.DataFrame(real_exp['real_signals'], columns=['Token', 'Weight']).sort_values('Weight', ascending=True)
axes[1].barh(real_df['Token'], real_df['Weight'], color='#10b981')
axes[1].set_title("🟢 Signals Pushing Towards 'Real' (Institutional Credibility)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Attribution Weight")

plt.tight_layout()
plt.show()
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## 4. Probability Calibration Analysis: Reliability Diagram & Brier Score"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""data_path = "../data/processed/cleaned_news_11k.csv"
if not os.path.exists(data_path):
    data_path = "data/processed/cleaned_news_11k.csv"

df = pd.read_csv(data_path)
X_test_sample = df.sample(n=2000, random_state=42)
X_vec = vectorizer.transform(X_test_sample['text'].tolist())
y_test = X_test_sample['label'].values
y_probs = model.predict_proba(X_vec)[:, 1]

prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10, strategy='uniform')

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.plot(prob_pred, prob_true, marker='o', color='#3b82f6', linewidth=2, label='Calibrated Production Model')
plt.title("Calibration Curve (Reliability Diagram)", fontsize=12, fontweight='bold')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (Actual Frequency)")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## 5. ❌ Systematic Error Analysis: Where Does the Model Fail?"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""err_path = "../data/processed/error_analysis.json"
if not os.path.exists(err_path):
    err_path = "data/processed/error_analysis.json"

with open(err_path, "r", encoding="utf-8") as f:
    error_data = json.load(f)

failures = pd.DataFrame(error_data['curated_failure_categories'])

for _, row in failures.iterrows():
    print(f"\\n❌ Failure Category: {row['category']}")
    print(f"   Example: \\"{row['example_snippet']}\\"")
    print(f"   Ground Truth: {row['ground_truth']} | Prediction: {row['model_prediction']}")
    print(f"   Root Cause: {row['failure_root_cause']}")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### Analytical Synthesis: Why Linguistic Errors Occur

- **Short Articles:** The model struggled with short articles because TF-IDF had insufficient textual information to distinguish their linguistic patterns from clickbait snippets.
- **Satire:** Satirical writing frequently adopts dry journalistic syntax, which fools superficial unigram frequencies.
- **Clickbait Discrepancies:** Headline sensation vs neutral body creates divergent signals that sparse bag-of-words representations cannot resolve without hierarchical attention.
"""
    ))

    nb['cells'] = cells
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    return output_path

if __name__ == "__main__":
    p = create_explainability_notebook()
    print("Explainability notebook created at:", p)
