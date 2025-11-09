"""Streamlit UI prototype for the Taglish Review Analyzer."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from importlib import util as importlib_util
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_PATH = PROJECT_ROOT / "app" / "inference" / "predictor.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

_spec = importlib_util.spec_from_file_location("app.inference.predictor", str(PREDICTOR_PATH))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load predictor module from {PREDICTOR_PATH}")
_module = importlib_util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

from app.inference.predictor import ReviewAnalyzer

API_URL = os.getenv("REVIEW_API_URL")
SAMPLE_REVIEWS: List[Tuple[str, str]] = [
    (
        "Positive Unboxing",
        "Super bilis ng delivery! Maayos ang packaging at may freebie pa si seller. Highly recommended!",
    ),
    (
        "Sizing Issue",
        "Maganda sana yung tela pero sobrang luwag ng size kahit medium ang inorder ko. Sana i-double check nila next time.",
    ),
    (
        "Delayed Delivery",
        "Ang tagal dumating ng parcel, halos dalawang linggo. Hindi rin nagrereply si seller sa follow-up ko.",
    ),
    (
        "Emoji Praise",
        "Solid na solid yung shop 😍🔥 mabilis shipping tapos super protected ang box 📦💯",
    ),
    (
        "Suspicious Review",
        "Best product ever!!! Buy now!!! Use my link: http://bit.ly/fREe-Deal para sa discount!!!",
    ),
]

ASPECT_COLORS = {
    "positive": "#8BC34A",
    "neutral": "#FFC107",
    "negative": "#F44336",
}


@st.cache_resource(show_spinner=False)
def get_local_analyzer() -> ReviewAnalyzer:
    return ReviewAnalyzer()


def analyze_local(review: str) -> Dict:
    analyzer = get_local_analyzer()
    return analyzer.analyze(review).to_dict()


def analyze_via_api(review: str) -> Dict:
    if not API_URL:
        raise RuntimeError("REVIEW_API_URL is not set.")
    response = requests.post(
        f"{API_URL.rstrip('/')}/analyze",
        json={"review": review},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def highlight_evidence(review: str, aspects: Iterable[Dict]) -> str:
    tokens = []
    for aspect in aspects:
        evidence = aspect.get("evidence") or ""
        tokens.extend([tok.strip() for tok in evidence.split(",") if tok.strip()])
    highlighted = review
    # Longer tokens first to avoid partial replacements
    for token in sorted(set(tokens), key=len, reverse=True):
        pattern = re.compile(re.escape(token), flags=re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", highlighted)
    return highlighted


def sentiment_confidence(sentiment: Dict) -> Tuple[str, float]:
    label = sentiment.get("label", "unknown").title()
    scores = sentiment.get("scores", {})
    prob = scores.get(sentiment.get("label", ""), 0.0)
    return label, float(prob)


def summarize_aspects(aspects: List[Dict]) -> Tuple[str, float]:
    if not aspects:
        return "None detected", 0.0
    sentiment_counter = Counter(entry.get("sentiment", "neutral") for entry in aspects)
    dominant = sentiment_counter.most_common(1)[0][0]
    return dominant.title(), sentiment_counter[dominant] / max(1, len(aspects))


def batch_analyze(reviews: Iterable[str], use_api: bool) -> List[Dict]:
    results = []
    for text in reviews:
        text = str(text)
        if not text.strip():
            continue
        analysis = analyze_via_api(text) if use_api else analyze_local(text)
        analysis["review"] = text
        results.append(analysis)
    return results


def build_batch_report(batch: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    aspect_counter = Counter()
    deception_scores = []
    for result in batch:
        sentiment = result.get("sentiment", {})
        label = sentiment.get("label", "unknown")
        conf = sentiment.get("scores", {}).get(label, 0.0)
        rows.append(
            {
                "review": result.get("review", ""),
                "sentiment": label,
                "confidence": conf,
                "deception": result.get("deception", {}).get("score", 0.0),
                "aspects": ", ".join(
                    f"{a.get('aspect','')}: {a.get('sentiment','') }" for a in result.get("aspects", [])
                ),
            }
        )
        for aspect in result.get("aspects", []):
            aspect_counter[aspect.get("aspect", "")] += 1
        deception_scores.append(result.get("deception", {}).get("score", 0.0))
    df = pd.DataFrame(rows)
    summary = {
        "reviews_analyzed": len(batch),
        "positive_share": float((df["sentiment"] == "positive").mean()) if not df.empty else 0.0,
        "negative_share": float((df["sentiment"] == "negative").mean()) if not df.empty else 0.0,
        "avg_deception": float(pd.Series(deception_scores).mean()) if deception_scores else 0.0,
        "top_aspect": aspect_counter.most_common(1)[0][0] if aspect_counter else "None",
    }
    return df, summary


def render_confidence_bars(scores: Dict) -> None:
    if not scores:
        return
    st.write("Confidence by class")
    cols = st.columns(len(scores))
    for col, (label, prob) in zip(cols, scores.items()):
        col.metric(label.title(), f"{prob:.0%}")
        col.progress(min(max(prob, 0.0), 1.0))


def show_summary_cards(sentiment: Dict, aspects: List[Dict], deception: float) -> None:
    label, prob = sentiment_confidence(sentiment)
    dominant_aspect, aspect_share = summarize_aspects(aspects)
    cols = st.columns(3)
    cols[0].metric("Sentiment", label, f"Confidence {prob:.0%}")
    cols[1].metric("Dominant Aspect Sentiment", dominant_aspect, f"{aspect_share:.0%} of detected")
    cols[2].metric("Deception Score", f"{deception:.0%}")


def render_aspect_table(aspects: List[Dict]) -> None:
    if not aspects:
        st.info("No aspect keywords detected. Add more detail to the review text.")
        return
    df = pd.DataFrame(aspects)
    def color_row(row):
        color = ASPECT_COLORS.get(row.get("sentiment", "").lower(), "#607D8B")
        return [f"background-color: {color}22"] * len(row)
    st.dataframe(df.style.apply(color_row, axis=1))


def render_about_sidebar() -> None:
    with st.sidebar.expander("About this prototype", expanded=False):
        st.markdown(
            """
            **Pipeline recap**
            - Fine-tuned `xlm-roberta-base` on FiReCS product reviews (macro-F1 ≈ 0.81).
            - Aspect snippets scored via TF–IDF + Logistic Regression per retail category.
            - Lightweight deception classifier combines stylometric signals.

            **How to demo**
            1. Pick a sample review or paste your own.
            2. Hit **Analyze** to view sentiment, aspect, and deception insights.
            3. Optionally upload a CSV (`review` column) for batch reporting.
            """
        )


def main() -> None:
    st.set_page_config(page_title="Taglish Review Analyzer", layout="wide")
    st.title("Taglish Review Analyzer")
    st.caption("End-to-end sentiment, aspect, and deception insights for Taglish product feedback.")

    render_about_sidebar()

    if "review_text" not in st.session_state:
        st.session_state.review_text = SAMPLE_REVIEWS[0][1]
    st.session_state.setdefault("batch_df", None)
    st.session_state.setdefault("batch_summary", None)

    st.sidebar.subheader("Guided samples")
    sample_labels = [label for label, _ in SAMPLE_REVIEWS]
    selected = st.sidebar.selectbox("Choose a sample", sample_labels)
    if st.sidebar.button("Load sample"):
        st.session_state.review_text = dict(SAMPLE_REVIEWS)[selected]

    st.sidebar.divider()
    st.sidebar.subheader("Batch analysis")
    use_api_sidebar = st.sidebar.checkbox("Use backend API", value=bool(API_URL))
    uploaded = st.sidebar.file_uploader("Upload CSV with a `review` column", type=["csv"])
    if uploaded is not None:
        with st.spinner("Processing batch..."):
            df_uploaded = pd.read_csv(uploaded)
            if "review" not in df_uploaded.columns:
                st.sidebar.error("CSV must contain a `review` column.")
            else:
                batch_results = batch_analyze(df_uploaded["review"].tolist(), use_api_sidebar)
                batch_df, summary = build_batch_report(batch_results)
                st.session_state.batch_df = batch_df
                st.session_state.batch_summary = summary
                st.sidebar.success(
                    f"Analyzed {summary['reviews_analyzed']} reviews | Pos share {summary['positive_share']:.0%}"
                )
                st.sidebar.write(f"Top aspect mentioned: **{summary['top_aspect']}**")
                st.sidebar.write(f"Average deception score: {summary['avg_deception']:.2f}")
                st.sidebar.download_button(
                    label="Download batch results (CSV)",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_analysis.csv",
                    mime="text/csv",
                    key="sidebar_batch_download",
                )

    if st.session_state.batch_df is not None:
        st.subheader("Batch analysis results")
        summary = st.session_state.batch_summary or {}
        cols = st.columns(3)
        cols[0].metric("Reviews analyzed", summary.get("reviews_analyzed", 0))
        cols[1].metric("Positive share", f"{summary.get('positive_share', 0.0):.0%}")
        cols[2].metric("Avg deception", f"{summary.get('avg_deception', 0.0):.2f}")
        st.write(f"Top aspect mentioned: **{summary.get('top_aspect', 'None')}**")
        st.dataframe(st.session_state.batch_df)
        st.download_button(
            label="Download batch results (CSV)",
            data=st.session_state.batch_df.to_csv(index=False).encode("utf-8"),
            file_name="batch_analysis.csv",
            mime="text/csv",
            key="main_batch_download",
        )
        st.divider()

    review_text = st.text_area("Paste or edit a review", st.session_state.review_text, height=160, key="review_text")
    use_api = st.checkbox(
        "Use backend API (requires REVIEW_API_URL env var)", value=bool(API_URL), key="use_api_checkbox"
    )

    if st.button("Analyze", type="primary"):
        if not review_text.strip():
            st.warning("Please enter a review first.")
            st.stop()
        st.session_state.batch_df = None
        st.session_state.batch_summary = None
        with st.spinner("Analyzing..."):
            try:
                result = analyze_via_api(review_text) if use_api else analyze_local(review_text)
            except Exception as exc:  # pragma: no cover - runtime safeguard
                st.error(f"Analysis failed: {exc}")
                st.stop()
        st.session_state.last_result = result
        st.session_state.last_review = review_text

    if "last_result" not in st.session_state:
        st.info("Run an analysis to see detailed insights.")
        return

    result = st.session_state.last_result
    review_text = st.session_state.last_review

    sentiment = result.get("sentiment", {})
    aspects = result.get("aspects", [])
    deception = float(result.get("deception", {}).get("score", 0.0))

    show_summary_cards(sentiment, aspects, deception)
    render_confidence_bars(sentiment.get("scores", {}))

    st.subheader("Aspect Insights")
    render_aspect_table(aspects)

    st.subheader("Highlighted Review")
    highlighted = highlight_evidence(review_text, aspects)
    st.markdown(f"<div style='padding:12px;background:#f6f8fa;border-radius:6px;'>{highlighted}</div>", unsafe_allow_html=True)

    st.subheader("Deception Signal")
    st.progress(min(max(deception, 0.0), 1.0))
    st.write(f"Estimated suspiciousness: **{deception:.2f}**")

    st.subheader("Raw Output")
    st.code(json.dumps(result, indent=2, ensure_ascii=False))
    st.download_button(
        label="Download result (JSON)",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="single_review_analysis.json",
        mime="application/json",
    )

    if st.session_state.batch_df is not None:
        st.subheader("Batch Analysis Results")
        st.dataframe(st.session_state.batch_df)
        st.write(f"Top aspect mentioned: **{st.session_state.batch_summary['top_aspect']}**")
        st.write(f"Average deception score: {st.session_state.batch_summary['avg_deception']:.2f}")
        st.download_button(
            label="Download batch results (CSV)",
            data=st.session_state.batch_df.to_csv(index=False).encode("utf-8"),
            file_name="batch_analysis.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()


