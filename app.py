"""StarvoxLabs assessment: Customer Feedback Insights Analyzer.
Run from this directory: python -m streamlit run app.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from urllib.parse import quote as url_quote

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core import TOPICS, SENTIMENTS, load_csv, clean_reviews, week_starts
from src.intake import REVIEW, SUPPORT, combine_sources, ingest_feedback_csv
from src.models import ClassicClassifier, EmbeddingClassifier, possible_new_issues
from src.aspects import explain_review
from src.summaries import issue_groups, extractive_summary, cloud_llm_summary

ROOT = Path(__file__).resolve().parent

def get_hosted_llm_key() -> str:
    """Secrets are configured by the project owner, never entered by reviewers."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""

st.set_page_config(page_title="Customer Feedback Insights | StarvoxLabs", page_icon="📊", layout="wide")
st.title("Customer Feedback Insights Analyzer")


@st.cache_data(show_spinner=False)
def get_demo():
    return load_csv(ROOT / "data" / "reviews.csv")

@st.cache_data(show_spinner=False)
def get_demo_support():
    return load_csv(ROOT / "data" / "support_messages.csv")

@st.cache_data(show_spinner=False)
def get_training_data():
    reviews, _ = clean_reviews(load_csv(ROOT / "data" / "reviews.csv"))
    # Evaluation examples never enter the classifier training data, even in demo mode.
    evaluation = load_csv(ROOT / "data" / "evaluation_to_verify.csv")
    held_out = set(evaluation["text"].dropna().astype(str).str.strip().str.casefold())
    return reviews.loc[~reviews["text"].str.casefold().isin(held_out)].copy()

@st.cache_resource(show_spinner=False)
def fit_models(mode: str):
    train = get_training_data()
    if mode == "MiniLM embeddings":
        # Both classifiers share the same transformer encoder to conserve memory.
        from sentence_transformers import SentenceTransformer
        from src.models import EMBEDDING_NAME
        encoder = SentenceTransformer(EMBEDDING_NAME)
        topic = EmbeddingClassifier(encoder=encoder)
        sentiment = EmbeddingClassifier(encoder=encoder)
    else:
        topic, sentiment = ClassicClassifier(), ClassicClassifier()
    topic.fit(train["text"], train["topic"])
    sentiment.fit(train["text"], train["sentiment"])
    return topic, sentiment

def predict_reviews(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    topic_model, sentiment_model = fit_models(mode)
    predicted = df.copy()
    predicted["topic"] = topic_model.predict(predicted["text"])
    predicted["sentiment"] = sentiment_model.predict(predicted["text"])
    predicted["confidence"] = topic_model.confidence(predicted["text"]).round(3)
    details = [explain_review(t, topic, sentiment) for t, topic, sentiment in zip(predicted["text"], predicted["topic"], predicted["sentiment"])]
    predicted["topic"] = [d["topic"] for d in details]
    predicted["sentiment"] = [d["sentiment"] for d in details]
    predicted["mixed"] = [d["mixed"] for d in details]
    predicted["explanation"] = [d["explanation"] for d in details]
    return predicted

with st.sidebar:
    st.header("Data and filters")
    feedback_upload = st.file_uploader("Upload Feedback File", type="csv", key="feedback", help="One CSV containing customer reviews or support messages. For a mixed file, add source_type = Review or Support message. Agent/bot replies are excluded when sender_role is supplied.")
    st.caption("Accepts review CSVs, support-message CSVs, or a combined CSV with source_type. With no upload, synthetic demo feedback is used. Public or synthetic data only.")
    model_mode = st.radio("Classification method", ["TF-IDF baseline", "MiniLM embeddings"], help="MiniLM needs optional installation and a first-time model download.")
    source_filter = st.multiselect("Feedback source", [REVIEW, SUPPORT], default=[REVIEW, SUPPORT])
    topic_filter = st.multiselect("Topics", TOPICS, default=TOPICS)
    sentiment_filter = st.multiselect("Sentiment", SENTIMENTS, default=SENTIMENTS)
    st.caption("Demo labels are synthetic. Dashboard predictions are not a substitute for held-out evaluation.")

try:
    if feedback_upload is None:
        cleaned, stats = combine_sources([(get_demo(), REVIEW), (get_demo_support(), SUPPORT)])
    else:
        cleaned, stats = ingest_feedback_csv(load_csv(feedback_upload))
    with st.spinner("Classifying customer feedback..."):
        data = predict_reviews(cleaned, model_mode)
except Exception as exc:
    st.error("Could not use this file : The file is empty. Upload a CSV with a header row and at least one review .")
    st.stop()

# User-selected citation displays the exact source row at the top of the page.
selected_id = st.query_params.get("review_id", "")
if selected_id:
    selected = data.loc[data["id"].eq(selected_id)]
    if not selected.empty:
        r = selected.iloc[0]
        with st.container(border=True):
            st.markdown(f"**{r['source_type']} {r['source_id']}** · {r['topic']} · {r['sentiment']}")
            if r["source_type"] == SUPPORT:
                st.caption(f"Ticket: {r['ticket_id'] or 'not supplied'} · Channel: {r['channel'] or 'not supplied'}")
            st.write(r["raw_text"])
            if r["explanation"]:
                st.caption(r["explanation"])

filtered = data.loc[data["source_type"].isin(source_filter) & data["topic"].isin(topic_filter) & data["sentiment"].isin(sentiment_filter)].copy()
valid_weeks = sorted(week_starts(filtered.loc[filtered["date"].notna(), "date"]).unique(), reverse=True)
period_options = ["All dates"] + [pd.Timestamp(w).strftime("%d %b %Y") for w in valid_weeks]
with st.sidebar:
    period = st.selectbox("Summary week (starts Monday)", period_options, index=1 if len(period_options)>1 else 0)

summary_df = filtered
if period != "All dates":
    selected_week = pd.to_datetime(period, format="%d %b %Y")
    summary_df = filtered.loc[week_starts(filtered["date"]) == selected_week].copy()

st.markdown("##### Data quality")
a, b, c, d, e, f = st.columns(6)
a.metric("CSV rows", stats["raw"])
b.metric("Usable items", stats["usable"])
c.metric("Duplicates removed", stats["duplicates"])
d.metric("Invalid text rows", stats["invalid"])
e.metric("Missing dates", stats["missing_dates"])
f.metric("Agent/bot replies excluded", stats["skipped_non_customer"])

st.markdown("##### Filtered results")
a, b, c, d, e, f = st.columns(6)
a.metric("Feedback items", len(filtered))
b.metric("Reviews", int(filtered["source_type"].eq(REVIEW).sum()))
c.metric("Support messages", int(filtered["source_type"].eq(SUPPORT).sum()))
d.metric("Negative", int(filtered["sentiment"].eq("negative").sum()))
e.metric("Urgent flags", int(filtered["urgent"].sum()))
f.metric("Mixed feedback", int(filtered["mixed"].sum()))

tab_overview, tab_evidence, tab_emerging, tab_evaluation = st.tabs(["Dashboard", "Weekly issues & quotes", "Possible new issues", "Model evaluation"])

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Topics")
        count_table = filtered["topic"].value_counts().rename_axis("Topic").reset_index(name="Reviews")
        if not count_table.empty:
            st.plotly_chart(px.bar(count_table, x="Topic", y="Reviews", text="Reviews"), use_container_width=True)
        else:
            st.info("No data for the selected filters.")
    with right:
        st.subheader("Sentiment")
        if not filtered.empty:
            st.plotly_chart(px.pie(filtered, names="sentiment", hole=0.45), use_container_width=True)
        else:
            st.info("No data for the selected filters.")
    st.subheader("Weekly sentiment trend")
    dated = filtered.dropna(subset=["date"]).copy()
    if not dated.empty:
        dated["week"] = week_starts(dated["date"])
        trend = dated.groupby(["week", "sentiment"]).size().reset_index(name="reviews")
        st.plotly_chart(px.line(trend, x="week", y="reviews", color="sentiment", markers=True), use_container_width=True)
    else:
        st.info("Provide a 'date' column to show the weekly trend.")
    st.subheader("Explore source feedback")
    columns = ["id", "source_type", "source_id", "ticket_id", "channel", "date", "topic", "sentiment", "urgent", "mixed", "confidence", "raw_text", "explanation"]
    st.dataframe(filtered[columns], hide_index=True, use_container_width=True)
    st.download_button("Export filtered feedback predictions", filtered[columns].to_csv(index=False).encode("utf-8"), "feedback_predictions.csv", "text/csv")

with tab_evidence:
    st.subheader(f"Top customer complaints · {period}")
    groups = issue_groups(summary_df)
    if not groups:
        st.success("No negative reviews in this selection. Positive reviews are not counted as complaints.")
    else:
        st.caption("Counts are computed from negative customer feedback only. Every claim links to source review/message IDs and verbatim quotes.")
        if "llm_result" not in st.session_state:
            st.session_state["llm_result"] = None
        llm_key = get_hosted_llm_key()
        if not llm_key:
            st.info("LLM generation is not configured by the app owner yet. Showing evidence-only summaries. On a hosted demo, set GROQ_API_KEY in Streamlit Cloud secrets.")
        if st.button("Generate LLM weekly summary", type="primary", disabled=not bool(llm_key)):
            with st.spinner("Generating summary and checking source citations..."):
                try:
                    items, accepted = cloud_llm_summary(groups, api_key=llm_key)
                    st.session_state["llm_result"] = {"key": [(g["topic"],g["count"],tuple((e["id"], e["quote"]) for e in g["evidence"])) for g in groups], "items": items}
                    st.success(f"LLM completed. {accepted}/{len(groups)} generated claims passed strict evidence checks; remaining topics use extractive evidence.")
                except RuntimeError as exc:
                    st.session_state["llm_result"] = None
                    st.warning(str(exc))
        signature = [(g["topic"],g["count"],tuple((e["id"], e["quote"]) for e in g["evidence"])) for g in groups]
        stored = st.session_state.get("llm_result")
        report = stored["items"] if stored and stored["key"] == signature else extractive_summary(groups)
        st.caption("Summary mode: " + ("hosted LLM with validated evidence (extractive fallback when needed)" if stored and stored["key"] == signature else "evidence-only extractive preview; click Generate to run the hosted LLM"))
        for item in report:
            with st.container(border=True):
                st.markdown(f"**{item['topic']} — {item['count']} negative feedback items**")
                st.write(item["headline"])
                st.caption("Evidence · click an ID to inspect the original review or support message")
                for evidence in item["evidence"]:
                    ref = url_quote(evidence["id"], safe="")
                    st.markdown(f"[{evidence['id']}](?review_id={ref})")
                    st.write(f"“{evidence['quote']}”")
                    if evidence.get("source_type") == SUPPORT:
                        st.caption(f"Support ticket: {evidence.get('ticket_id') or 'not supplied'} · Channel: {evidence.get('channel') or 'not supplied'}")
        st.caption("Important: A source ID proves a quoted review exists; a person should still review nuanced interpretations before acting on them.")

with tab_emerging:
    st.subheader("Candidate new issues")
    st.caption("K-Means groups feedback labelled 'Other' or having low confidence. These are exploratory leads, not confirmed new categories.")
    cutoff = st.slider("Low-confidence threshold", 0.25, 0.90, 0.55, 0.05)
    clusters = possible_new_issues(data, threshold=cutoff)
    if not clusters:
        st.info("Not enough 'Other' or low-confidence records to form meaningful clusters at this threshold.")
    for cluster in clusters:
        with st.expander(f"Candidate cluster {cluster['group']} · {cluster['count']} reviews"):
            for example in cluster["examples"]:
                st.markdown(f"**[{example['id']}](?review_id={url_quote(str(example['id']),safe='')})**")
                st.write(example["raw_text"])

with tab_evaluation:
    st.subheader("Held-out model comparison")
    path = ROOT / "outputs" / "evaluation_results.json"
    if not path.exists():
        st.warning("Official results pending: hand-check at least 100 rows in data/evaluation_to_verify.csv, mark human_verified=yes, and run python -m src.evaluate.")
        st.code("python -m pip install -r requirements-embeddings.txt\npython -m src.evaluate", language="powershell")
    else:
        result = json.loads(path.read_text(encoding="utf-8"))
        entries = [{"task / model": name, "accuracy": details["accuracy"], "macro F1": details["macro_f1"]} for name, details in result["scores"].items()]
        st.caption(f"Hand-verified held-out samples: {result['verified_rows']} · Training rows: {result['training_rows']}. All methods evaluated using the same IDs.")
        st.dataframe(pd.DataFrame(entries), hide_index=True, use_container_width=True)
        chosen = st.selectbox("View confusion matrix", list(result["scores"]))
        matrix = result["scores"][chosen]
        st.plotly_chart(px.imshow(matrix["confusion_matrix"], x=matrix["labels"], y=matrix["labels"], text_auto=True, labels={"x":"Predicted","y":"Actual"}), use_container_width=True)
        st.caption("For five analysed errors, complete the why_the_model_failed column in outputs/mistakes_to_analyse.csv.")
