import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.calibration import format_prediction_result, get_confidence_tier
from src.feature_extraction import get_tfidf_vectorizer
from src.evaluation import evaluate_model_performance
from src.explainability import ModelExplainer
from sklearn.linear_model import LogisticRegression

class TestFakeNewsPipeline(unittest.TestCase):
    def test_calibration_formatting(self):
        res_fake = format_prediction_result(0.92)
        self.assertTrue(res_fake["is_fake"])
        self.assertEqual(res_fake["confidence"], "High")
        self.assertEqual(res_fake["probability_fake"], 92.0)
        
        res_real = format_prediction_result(0.12)
        self.assertFalse(res_real["is_fake"])
        self.assertEqual(res_real["confidence"], "High")
        self.assertEqual(res_real["probability_real"], 88.0)
        
        res_mid = format_prediction_result(0.55)
        self.assertEqual(res_mid["confidence"], "Low")

    def test_evaluation_metrics(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.85])
        
        metrics = evaluate_model_performance(y_true, y_pred, y_prob, model_name="TestModel")
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("brier_score", metrics)
        self.assertIn("reasoning", metrics)

    def test_tfidf_and_explainability(self):
        corpus = [
            "WASHINGTON (Reuters) - Government officials confirmed fiscal data report today in press room.",
            "BREAKING NEWS: Shocking revelation shows secret alien bunker confirmed in viral report!",
            "Federal bank statements verify inflation numbers according to treasury officials."
        ]
        labels = np.array([0, 1, 0])
        vec = get_tfidf_vectorizer(max_features=100, min_df=1, max_df=1.0)
        X = vec.fit_transform(corpus)
        clf = LogisticRegression()
        clf.fit(X, labels)
        
        explainer = ModelExplainer(vec, clf)
        exp = explainer.explain_fast_attribution("Shocking breaking news exposed!", top_k=3)
        self.assertIn("fake_signals", exp)
        self.assertIn("real_signals", exp)
        self.assertIn("key_indicators", exp)

if __name__ == "__main__":
    unittest.main()
