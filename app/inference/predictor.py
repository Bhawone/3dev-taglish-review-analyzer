"""
Shared inference utilities for the Taglish review analyzer.

The module attempts to load the fine-tuned transformer checkpoint stored in
`checkpoints/xlmrb/best`. If it cannot be found, it falls back to a
TF-IDF + Logistic Regression baseline that is trained on demand using the
FiReCS dataset and persisted split IDs.

Aspect detection and deception scoring are currently heuristic-based
placeholders until the dedicated models are trained.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - deployed environments without torch
    torch = None
try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - deployed environments without joblib
    joblib = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The transformers package is required for predictor. "
        "Install it via `pip install transformers`."
    ) from exc


# Get project root (parent of app directory)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints" / "xlmrb" / "best"
MODELS_DIR = _PROJECT_ROOT / "models"

# Try multiple locations for FiReCS.csv
_firecs_candidates = [
    _PROJECT_ROOT / "FiReCS.csv",
    _PROJECT_ROOT / "Final_Project_Deliverables" / "FiReCS.csv",
]
FIRECS_PATH = None
for candidate in _firecs_candidates:
    if candidate.exists():
        FIRECS_PATH = candidate
        break
if FIRECS_PATH is None:
    # Use root as default (will raise error if not found when needed)
    FIRECS_PATH = _PROJECT_ROOT / "FiReCS.csv"

TRAIN_IDS_PATH = _PROJECT_ROOT / "train_ids.csv"
VAL_IDS_PATH = _PROJECT_ROOT / "val_ids.csv"
TEST_IDS_PATH = _PROJECT_ROOT / "test_ids.csv"

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


ASPECT_KEYWORDS = {
    "product_quality": [
        "quality",
        "matibay",
        "sira",
        "defect",
        "fabric",
        "built",
        "pangit",
        "ayos",
    ],
    "sizing_fit": ["size", "fit", "laki", "liit", "sakto", "oversize"],
    "delivery_courier": ["deliver", "lalamove", "courier", "ship", "delay", "on time"],
    "packaging": ["package", "box", "bubble", "sealed", "damaged"],
    "price_value": ["price", "sulit", "worth", "mura", "mahal"],
    "seller_communication": ["seller", "reply", "response", "support", "chat"],
}

POSITIVE_CUES = {"ganda", "ayos", "sulit", "ok", "legit", "maganda", "masarap"}
NEGATIVE_CUES = {"pangit", "bad", "sira", "late", "scam", "fake", "bitin"}


@dataclass
class AnalysisResult:
    sentiment_label: str
    sentiment_scores: Dict[str, float]
    aspects: List[Dict[str, str]]
    deception_score: float

    def to_dict(self) -> Dict:
        top_label = self.sentiment_label
        top_score = self.sentiment_scores.get(top_label, 0.0)
        return {
            "sentiment": {
                "label": self.sentiment_label,
                "scores": self.sentiment_scores,
                "message": f"Overall sentiment is {top_label} with confidence {top_score:.2%}.",
            },
            "aspects": self.aspects,
            "deception": {"score": self.deception_score},
        }


class ReviewAnalyzer:
    def __init__(self) -> None:
        self.device = (
            torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
            if torch is not None
            else None
        )
        self.transformer_tokenizer: Optional[AutoTokenizer] = None
        self.transformer_model: Optional[AutoModelForSequenceClassification] = None
        self.baseline_pipeline: Optional[Pipeline] = None
        self.aspect_sentiment_models: Dict[str, Pipeline] = {}
        self.deception_model: Optional[LogisticRegression] = None

        if self._transformer_available():
            self._load_transformer()
        else:
            if torch is None:
                print("torch not available; using baseline pipeline only.")
            else:
                print("Transformer checkpoint not found; training fallback baseline.")
            self._train_baseline()
        self._load_auxiliary_models()

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #
    def _transformer_available(self) -> bool:
        if torch is None:
            return False
        if not CHECKPOINT_DIR.exists():
            return False
        required_files = ["config.json", "tokenizer.json", "sentencepiece.bpe.model"]
        model_files = {"pytorch_model.bin", "model.safetensors", "pytorch_model_quantized.bin"}
        has_required = all((CHECKPOINT_DIR / fname).exists() for fname in required_files)
        has_weights = any((CHECKPOINT_DIR / fname).exists() for fname in model_files)
        return has_required and has_weights

    def _load_transformer(self) -> None:
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINT_DIR, use_fast=True
        )
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            CHECKPOINT_DIR
        ).to(self.device)
        self.transformer_model.eval()

    def _load_auxiliary_models(self) -> None:
        if joblib is None or not MODELS_DIR.exists():
            return
        aspect_path = MODELS_DIR / "aspect_sentiment_models.joblib"
        if aspect_path.exists():
            try:
                self.aspect_sentiment_models = joblib.load(aspect_path)
            except Exception as exc:  # pragma: no cover
                print(f"Failed to load aspect sentiment models: {exc}")
        deception_path = MODELS_DIR / "deception_classifier.joblib"
        if deception_path.exists():
            try:
                self.deception_model = joblib.load(deception_path)
            except Exception as exc:  # pragma: no cover
                print(f"Failed to load deception classifier: {exc}")

    def _train_baseline(self) -> None:
        if not FIRECS_PATH.exists():
            raise FileNotFoundError(
                "FiReCS.csv is required to train the fallback baseline."
            )
        df = pd.read_csv(FIRECS_PATH)
        df.columns = [c.lower() for c in df.columns]
        if not {"review", "label"} <= set(df.columns):
            raise ValueError("FiReCS.csv must contain 'review' and 'label' columns.")

        if TRAIN_IDS_PATH.exists() and "id" in df.columns:
            train_ids = pd.read_csv(TRAIN_IDS_PATH)["id"].tolist()
            train_df = df[df["id"].isin(train_ids)]
        else:
            train_df = df

        self.baseline_pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=3,
                        max_df=0.9,
                        lowercase=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        multi_class="auto",
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self.baseline_pipeline.fit(train_df["review"].astype(str), train_df["label"])

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _predict_transformer(self, text: str) -> Dict[str, float]:
        assert self.transformer_model and self.transformer_tokenizer
        enc = self.transformer_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            logits = self.transformer_model(**enc).logits.softmax(dim=-1).cpu().numpy()[0]
        return {LABEL_MAP[i]: float(logits[i]) for i in range(len(LABEL_MAP))}

    def _predict_baseline(self, text: str) -> Dict[str, float]:
        assert self.baseline_pipeline
        probs = self.baseline_pipeline.predict_proba([text])[0]
        return {LABEL_MAP[i]: float(probs[i]) for i in range(len(LABEL_MAP))}

    def _detect_aspects(self, text: str, sentiment_label: str) -> List[Dict[str, str]]:
        text_lower = text.lower()
        aspects = []
        for aspect, keywords in ASPECT_KEYWORDS.items():
            hits = [kw for kw in keywords if kw in text_lower]
            if hits:
                entry = {
                    "aspect": aspect,
                    "evidence": ", ".join(hits[:3]),
                    "sentiment": self._heuristic_aspect_sentiment(text_lower, hits, sentiment_label),
                }
                entry.setdefault("confidence", None)
                entry = self._predict_aspect_sentiment(text, entry)
                aspects.append(entry)
        return aspects

    def _heuristic_aspect_sentiment(
        self, text_lower: str, hits: List[str], default_label: str
    ) -> str:
        score = 0
        for cue in POSITIVE_CUES:
            if cue in text_lower:
                score += 1
        for cue in NEGATIVE_CUES:
            if cue in text_lower:
                score -= 1

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return default_label

    def _predict_aspect_sentiment(self, text: str, aspect_entry: Dict[str, str]) -> Dict[str, str]:
        if not self.aspect_sentiment_models:
            return aspect_entry
        model = self.aspect_sentiment_models.get(aspect_entry["aspect"])
        if model is None:
            return aspect_entry
        snippet = text
        evidence = aspect_entry.get("evidence")
        if evidence:
            first_kw = evidence.split(",")[0].strip()
            idx = text.lower().find(first_kw.lower())
            if idx != -1:
                start = max(0, idx - 60)
                end = min(len(text), idx + len(first_kw) + 60)
                snippet = text[start:end]
        proba = model.predict_proba([snippet])[0]
        best_idx = int(np.argmax(proba))
        aspect_entry["sentiment"] = SENTIMENT_LABELS[best_idx]
        aspect_entry["confidence"] = float(np.max(proba))
        return aspect_entry

    @staticmethod
    def _deception_feature_vector(text: str) -> np.ndarray:
        length = len(text)
        upper_ratio = sum(ch.isupper() for ch in text) / max(1, length)
        digit_ratio = sum(ch.isdigit() for ch in text) / max(1, length)
        punct_ratio = sum(ch in "!?" for ch in text) / max(1, length)
        repeat_chars = len(re.findall(r"(.)\\1{3,}", text.lower()))
        url_tokens = len(re.findall(r"http|www\\.|\\.com", text.lower()))
        exclaim = text.count("!")
        return np.array([[length, upper_ratio, digit_ratio, punct_ratio, repeat_chars, url_tokens, exclaim]])

    def _deception_score(self, text: str) -> float:
        length = len(text)
        uppercase_ratio = sum(ch.isupper() for ch in text) / max(length, 1)
        repeats = len(re.findall(r"(.)\\1{3,}", text.lower()))
        link_tokens = len(re.findall(r"(http|www\\.|\\.com)", text.lower()))
        score = 0.0
        score += min(repeats * 0.15, 0.45)
        score += min(link_tokens * 0.2, 0.4)
        score += 0.2 if uppercase_ratio > 0.4 else 0.0
        score += 0.1 if length < 20 else 0.0
        heuristic = max(0.0, min(score, 1.0))
        if self.deception_model is not None:
            proba = float(self.deception_model.predict_proba(self._deception_feature_vector(text))[0][1])
            return max(heuristic, proba)
        return heuristic

    def analyze(self, text: str) -> AnalysisResult:
        if not text.strip():
            raise ValueError("Input text must be non-empty.")

        raw_scores = (
            self._predict_transformer(text)
            if self.transformer_model is not None
            else self._predict_baseline(text)
        )
        sentiment_label = max(raw_scores, key=raw_scores.get)

        aspects = self._detect_aspects(text, sentiment_label)
        sentiment_label, scores = self._resolve_overall_sentiment(
            sentiment_label, raw_scores, aspects, text
        )
        deception = self._deception_score(text)

        return AnalysisResult(sentiment_label, scores, aspects, deception)

    def _resolve_overall_sentiment(
        self,
        initial_label: str,
        scores: Dict[str, float],
        aspects: List[Dict[str, str]],
        text: str,
    ) -> tuple[str, Dict[str, float]]:
        adjusted_scores = dict(scores)
        aspect_sentiments = [
            (a.get("sentiment") or "").lower()
            for a in aspects
            if a.get("sentiment") in {"positive", "neutral", "negative"}
        ]
        if aspect_sentiments:
            sentiment_counter = Counter(aspect_sentiments)
            pos_weight = sum(
                (a.get("confidence") or 0.35)
                for a in aspects
                if (a.get("sentiment") or "").lower() == "positive"
            )
            neg_weight = sum(
                (a.get("confidence") or 0.35)
                for a in aspects
                if (a.get("sentiment") or "").lower() == "negative"
            )
            has_pos = sentiment_counter.get("positive", 0) > 0
            has_neg = sentiment_counter.get("negative", 0) > 0
            dominant_sentiment, dominant_count = sentiment_counter.most_common(1)[0]
            aspect_total = sum(sentiment_counter.values())
            dominant_ratio = dominant_count / max(1, aspect_total)
            opposes_overall = (
                (initial_label == "positive" and dominant_sentiment == "negative")
                or (initial_label == "negative" and dominant_sentiment == "positive")
            )
            weighted_opposition = (
                (initial_label == "positive" and neg_weight > max(pos_weight, 0.4))
                or (initial_label == "negative" and pos_weight > max(neg_weight, 0.4))
            )
            if has_pos and has_neg:
                should_neutral = True
            elif opposes_overall and dominant_ratio >= 0.5:
                should_neutral = True
            elif weighted_opposition:
                should_neutral = True
            else:
                should_neutral = False
        else:
            should_neutral = False

        if should_neutral:
            neutral_target = max(
                adjusted_scores.get("neutral", 0.0),
                min(
                    0.9,
                    (adjusted_scores.get("positive", 0.0) + adjusted_scores.get("negative", 0.0))
                    / 2
                    + 0.1,
                ),
            )
            adjusted_scores["neutral"] = neutral_target
            adjusted_scores["positive"] = min(
                adjusted_scores.get("positive", 0.0), neutral_target * 0.85
            )
            adjusted_scores["negative"] = min(
                adjusted_scores.get("negative", 0.0), neutral_target * 0.85
            )
            return "neutral", adjusted_scores

        text_lower = text.lower()
        has_pos = any(cue in text_lower for cue in POSITIVE_CUES)
        has_neg = any(cue in text_lower for cue in NEGATIVE_CUES)
        if has_pos and has_neg:
            neutral_target = max(adjusted_scores.get("neutral", 0.0), 0.6)
            adjusted_scores["neutral"] = neutral_target
            adjusted_scores["positive"] = min(
                adjusted_scores.get("positive", 0.0), neutral_target * 0.9
            )
            adjusted_scores["negative"] = min(
                adjusted_scores.get("negative", 0.0), neutral_target * 0.9
            )
            return "neutral", adjusted_scores

        return initial_label, adjusted_scores


def analyze_text(text: str) -> Dict:
    global _ANALYZER
    try:
        analyzer = _ANALYZER
    except NameError:
        _ANALYZER = ReviewAnalyzer()
        analyzer = _ANALYZER
    return analyzer.analyze(text).to_dict()


if __name__ == "__main__":
    sample_text = (
        "Solid yung quality pero medyo delay ang delivery. "
        "Seller nagreply agad kaya ok pa rin."
    )
    result = analyze_text(sample_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))


