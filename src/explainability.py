import re
import numpy as np
from lime.lime_text import LimeTextExplainer

class ModelExplainer:
    def __init__(self, vectorizer, model, class_names=None):
        self.vectorizer = vectorizer
        self.model = model
        self.class_names = class_names or ["Real", "Fake"]
        self.lime_explainer = LimeTextExplainer(class_names=self.class_names)

    def explain_with_lime(self, text, num_features=6):
        def predict_fn(texts):
            vecs = self.vectorizer.transform(texts)
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(vecs)
            decision = self.model.decision_function(vecs)
            prob_fake = 1 / (1 + np.exp(-decision))
            return np.vstack([1 - prob_fake, prob_fake]).T

        exp = self.lime_explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_fn,
            num_features=num_features,
            labels=[1]
        )
        return exp.as_list(label=1)

    def explain_fast_attribution(self, text, top_k=5):
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        if not tokens:
            return {"fake_signals": [], "real_signals": []}

        feature_names = self.vectorizer.get_feature_names_out()
        vocab_index = self.vectorizer.vocabulary_

        weights = None
        base_estimator = self.model
        if hasattr(self.model, "calibrated_classifiers_"):
            base_estimator = self.model.calibrated_classifiers_[0].estimator

        if hasattr(base_estimator, "coef_"):
            weights = base_estimator.coef_[0]
        elif hasattr(base_estimator, "feature_importances_"):
            weights = base_estimator.feature_importances_

        if weights is None:
            return self._heuristic_attribution(tokens, top_k)

        signals = []
        seen = set()
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            if t in vocab_index:
                idx = vocab_index[t]
                w = float(weights[idx])
                signals.append((t, round(w, 4)))

        fake_signals = sorted([s for s in signals if s[1] > 0], key=lambda x: x[1], reverse=True)[:top_k]
        real_signals = sorted([s for s in signals if s[1] < 0], key=lambda x: x[1])[:top_k]
        real_signals_pos = [(w, round(abs(score), 4)) for w, score in real_signals]

        return {
            "fake_signals": fake_signals,
            "real_signals": real_signals_pos,
            "key_indicators": self._detect_heuristics(text)
        }

    def _heuristic_attribution(self, tokens, top_k=5):
        fake_lexicon = {"breaking", "shocking", "bombshell", "unbelievable", "revealed", "secret", "exposed", "hoax", "viral", "conspiracy"}
        real_lexicon = {"reuters", "confirmed", "department", "official", "statement", "spokesman", "tuesday", "republican", "democratic", "federal"}
        
        fake_signals = [(t, 0.15) for t in tokens if t in fake_lexicon][:top_k]
        real_signals = [(t, 0.15) for t in tokens if t in real_lexicon][:top_k]
        return {
            "fake_signals": fake_signals,
            "real_signals": real_signals,
            "key_indicators": ["sensational vocabulary matches", "lexical style heuristics"]
        }

    def _detect_heuristics(self, text):
        indicators = []
        sensational_words = ["shocking", "breaking", "bombshell", "unbelievable", "panic", "miracle", "exposed", "stabs", "secret plot"]
        found_sensational = [w for w in sensational_words if re.search(r'\b' + re.escape(w) + r'\b', text, re.I)]
        if found_sensational:
            indicators.append(f"Sensational wording detected: {', '.join(found_sensational[:3])}")
        
        if text.count('!') > 2 or text.count('?') > 2:
            indicators.append("Elevated punctuation intensity (! / ?)")

        upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        if upper_ratio > 0.08:
            indicators.append("Unusual uppercase character ratio (> 8%)")

        if not re.search(r'\b(Reuters|Associated Press|BBC|CNN|Bloomberg)\b', text, re.I):
            indicators.append("Low institutional source reliability signal")
        else:
            indicators.append("Verified wire/institutional attribution markers found")

        return indicators
