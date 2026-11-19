"""Customer Feedback Insights Analyzer — compact deployable Streamlit MVP."""
from __future__ import annotations
import json
import logging
import hashlib
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
from src.language import language_flags, translate_for_analysis
from src.image_feedback import describe_image
from src.summaries import issue_groups, extractive_summary, cloud_llm_summary

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_ITEMS = 20000
PREDICTION_BATCH_SIZE = 400

def get_hosted_llm_key() -> str:
    """Secrets are configured by the project owner, never entered by reviewers."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""

@st.cache_data(show_spinner=False)
def get_business_rules() -> dict:
    """Optional approved policy for a deployment; empty by default."""
    path = ROOT / 'config' / 'business_rules.json'
    try:
        rules = json.loads(path.read_text(encoding='utf-8'))
        for key in ('known_issues', 'expected_variations'):
            if not isinstance(rules.get(key), list) or not all(isinstance(v, str) for v in rules[key]):
                raise ValueError('Business rule lists must contain strings')
        return rules
    except (OSError, ValueError, KeyError):
        LOG.warning('Could not read valid business rules, using empty defaults')
        return {'known_issues': [], 'expected_variations': []}

st.set_page_config(page_title="Customer Feedback Insights Analyzer", page_icon="📊", layout="wide")
st.title("Customer Feedback Insights Analyzer")
st.caption("Analyse customer reviews and support messages with source-linked evidence.")

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
    """Run inference in bounded batches. Train models once per Streamlit cache."""
    topic_model, sentiment_model = fit_models(mode)
    predicted = df.copy()
    topics, sentiments, confidences = [], [], []
    for start in range(0, len(predicted), PREDICTION_BATCH_SIZE):
        chunk = predicted["text"].iloc[start:start + PREDICTION_BATCH_SIZE]
        topics.extend(topic_model.predict(chunk))
        sentiments.extend(sentiment_model.predict(chunk))
        confidences.extend(topic_model.confidence(chunk))
    predicted["topic"] = topics
    predicted["sentiment"] = sentiments
    predicted["confidence"] = pd.Series(confidences, index=predicted.index).round(3)
    details = [explain_review(t, topic, sentiment) for t, topic, sentiment in
               zip(predicted["text"], predicted["topic"], predicted["sentiment"])]
    predicted["topic"] = [d["topic"] for d in details]
    predicted["sentiment"] = [d["sentiment"] for d in details]
    predicted["mixed"] = [d["mixed"] for d in details]
    predicted["explanation"] = [d["explanation"] for d in details]
    predicted["aspect_details"] = [" | ".join(
        f"{a['topic']}: {a['sentiment']}" for a in d["aspects"]
    ) for d in details]
    return predicted

with st.sidebar:
    st.header("Data and filters")
    feedback_upload = st.file_uploader("Upload Feedback File", type=["csv", "png", "jpg", "jpeg"], key="feedback", help="CSV reviews/support messages; optionally one image attachment for a human-confirmed description.")
    st.caption("Single CSV uploader for reviews/support messages, or one optional image. With no upload, synthetic demo data is used. Public or synthetic inputs only.")


    with st.sidebar.expander("Advanced: Model Comparison", expanded=False):
        model_mode = st.radio(
            "Classification method",
            ["MiniLM embeddings", "TF-IDF baseline"],
            index=0,
            help="Compare the two classification approaches.",
        )


    translate_non_english = st.checkbox("AI-translate detected Hindi/Hinglish (optional, up to 40 items)", value=False,
                                      help="Sends selected feedback to Groq. Original quotes are never replaced. Translation requires review.")
    source_filter = st.multiselect("Feedback source", [REVIEW, SUPPORT], default=[REVIEW, SUPPORT])
    topic_filter = st.multiselect("Topics", TOPICS, default=TOPICS)
    sentiment_filter = st.multiselect("Sentiment", SENTIMENTS, default=SENTIMENTS)
    if feedback_upload is None:
        st.caption("You're viewing sample data. Upload your own feedback to see your customer insights.")
image_mode = feedback_upload is not None and not feedback_upload.name.lower().endswith('.csv')
image_description = ''
if image_mode:
    image_bytes = feedback_upload.getvalue()
    digest = hashlib.sha256(image_bytes).hexdigest()[:16]
    st.sidebar.image(image_bytes, caption="Image attachment (optional)", width="stretch")
    st.sidebar.caption("A photo alone is not a customer's complaint. Confirm what it shows before classifying. Captions are NOT original customer quotes.")
    if st.sidebar.button("Describe image with AI (optional)"):
        try:
            st.session_state[f'image_caption_{digest}'] = describe_image(
                image_bytes, 'image/png' if feedback_upload.name.lower().endswith('.png') else 'image/jpeg',
                api_key=get_hosted_llm_key(),
            )
            st.session_state[f'image_text_{digest}'] = st.session_state[f'image_caption_{digest}']
        except (RuntimeError, ValueError) as exc:
            st.sidebar.warning(str(exc))
    image_description = st.sidebar.text_area(
        "Confirm or write a description", value=st.session_state.get(f'image_caption_{digest}', ''),
        key=f'image_text_{digest}', help="Write the customer's observed problem if known. AI cannot infer it from a photo alone.",
    ).strip()
    if not image_description:
        st.info("Enter or confirm an image description in the sidebar to analyse this attachment.")
        st.stop()

try:
    if feedback_upload is None:
        cleaned, stats = combine_sources([(get_demo(), REVIEW), (get_demo_support(), SUPPORT)])
    elif image_mode:
        cleaned, stats = ingest_feedback_csv(pd.DataFrame({"id": ["IMAGE-1"], "text": [image_description]}))
        cleaned["is_image_caption"] = True  # excluded from quote-backed summaries
    else:
        if getattr(feedback_upload, "size", 0) > MAX_UPLOAD_BYTES:
            raise ValueError("CSV is larger than 15 MB. Split your export into smaller files.")
        cleaned, stats = ingest_feedback_csv(load_csv(feedback_upload))
    if len(cleaned) > MAX_ITEMS:
        raise ValueError(f"This demo accepts up to {MAX_ITEMS:,} cleaned items per upload. Split larger exports.")
    cleaned["language_hint"] = language_flags(cleaned["raw_text"])
    cleaned["translation_status"] = "original"
    unusual = cleaned.loc[cleaned["language_hint"].ne("English/undetermined")]
    if translate_non_english and not unusual.empty:
        key = get_hosted_llm_key()
        if key:
            try:
                selection = unusual.head(40)
                translated = translate_for_analysis(selection[["id", "text"]], api_key=key)
                for row_id, english in translated.items():
                    cleaned.loc[cleaned["id"].eq(row_id), "text"] = english
                    cleaned.loc[cleaned["id"].eq(row_id), "translation_status"] = "AI-translated; verify"
            except RuntimeError:
                st.warning("Translation unavailable; showing predictions from original text. Non-English results need review.")
        else:
            st.warning("Optional translation needs GROQ_API_KEY in Streamlit secrets.")
    with st.spinner("Classifying customer feedback..."):
        data = predict_reviews(cleaned, model_mode)
except ValueError as exc:
    st.error(f"Invalid feedback CSV: {exc}")
    st.stop()
except MemoryError:
    st.error("Not enough memory to process this file. Try a smaller CSV.")
    st.stop()
except RuntimeError:
    LOG.exception("Classifier/model loading failure")
    st.error("The classification model could not be loaded. Check server dependencies and logs.")
    st.stop()
except Exception:
    LOG.exception("Unexpected feedback processing failure")
    st.error("Feedback processing failed. Check application logs and the CSV structure.")
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

st.markdown("##### Data quality")
a, b, c, d, e, f = st.columns(6)
a.metric("CSV rows", stats["raw"])
b.metric("Usable items", stats["usable"])
c.metric("Duplicates removed", stats["duplicates"])
d.metric("Invalid text rows", stats["invalid"])
e.metric("Missing dates", stats["missing_dates"])
f.metric("Agent/bot replies excluded", stats["skipped_non_customer"])
if data["language_hint"].ne("English/undetermined").any():
    st.info("Possible non-English feedback detected. English-trained classification may be inaccurate. AI translation is optional and should be checked against the original text.")

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
            st.plotly_chart(px.bar(count_table, x="Topic", y="Reviews", text="Reviews"), width="stretch")
        else:
            st.info("No data for the selected filters.")
    with right:
        st.subheader("Sentiment")
        if not filtered.empty:
            st.plotly_chart(px.pie(filtered, names="sentiment", hole=0.45), width="stretch")
        else:
            st.info("No data for the selected filters.")
    st.subheader("Weekly sentiment trend")
    dated = filtered.dropna(subset=["date"]).copy()
    if not dated.empty:
        dated["week"] = week_starts(dated["date"])
        trend = dated.groupby(["week", "sentiment"]).size().reset_index(name="reviews")
        st.plotly_chart(
    px.line(
        trend,
        x="week",
        y="reviews",
        color="sentiment",
        markers=True
    ),
    width="stretch",
    config={"displayModeBar": False}
)
    else:
        st.info("Provide a 'date' column to show the weekly trend.")
    st.subheader("Explore source feedback")
    columns = ["id", "source_type", "source_id", "ticket_id", "channel", "date", "topic", "sentiment", "urgent", "mixed", "confidence", "raw_text", "explanation", "aspect_details", "language_hint", "translation_status"]
    st.dataframe(filtered[columns], hide_index=True, width="stretch")
    st.download_button("Export filtered feedback predictions", filtered[columns].to_csv(index=False).encode("utf-8"), "feedback_predictions.csv", "text/csv")

with tab_evidence:
    # Keep the week selector next to the complaints heading (not in the sidebar).
    heading_col, week_col = st.columns([3, 1])

    with heading_col:
        st.subheader("Top Customer Complaints")

    with week_col:
        period = st.selectbox(
            "Choose week",
            period_options,
            index=1 if len(period_options) > 1 else 0,
            key="summary_week",
        )

    # Filter to the selected week before finding complaint groups.
    summary_df = filtered.copy()
    if period != "All dates":
        selected_week = pd.to_datetime(period, format="%d %b %Y")
        summary_df = filtered.loc[
            week_starts(filtered["date"]) == selected_week
        ].copy()

    # This must run for both a chosen week and "All dates".
    # Image captions are not original customer reviews, so exclude them.
    groups = [] if image_mode else issue_groups(summary_df)

    if image_mode:
        st.info("Image descriptions are not original customer quotes and are excluded from weekly summaries.")
    elif not groups:
        st.info("No negative customer complaints found for this selection.")
    else:
        st.caption(
            "Explore the most common customer complaints and view the original feedback behind each insight."
        )
        llm_key = get_hosted_llm_key()
        if not llm_key:
            st.info("AI summaries are currently unavailable. Please contact the app administrator.")

        # The signature prevents a generated summary from appearing for a different week.
        signature = [
            (
                g["topic"],
                g["count"],
                tuple((e["id"], e["quote"]) for e in g["evidence"]),
            )
            for g in groups
        ]

        if st.button("Generate AI weekly summary", type="primary", disabled=not bool(llm_key)):
            st.session_state["summary_requested_key"] = signature
            with st.spinner("Generating summary and verifying customer evidence..."):
                try:
                    items, accepted = cloud_llm_summary(groups, api_key=llm_key)
                    st.session_state["llm_result"] = {
                        "key": signature,
                        "items": items,
                    }
                    st.success(
                        f"AI summary generated. {accepted}/{len(groups)} insights passed evidence checks; "
                        "any remaining insights use verified source-based summaries."
                    )
                except RuntimeError as exc:
                    st.session_state["llm_result"] = None
                    st.warning(f"AI summary unavailable: {exc}")

        stored = st.session_state.get("llm_result")
        if stored and stored.get("key") == signature:
            report = stored["items"]
        elif st.session_state.get("summary_requested_key") == signature:
            # Show extractive evidence only AFTER the user requests the AI summary.
            report = extractive_summary(groups)
            st.info("AI summary unavailable. Showing a source-based summary instead.")
        else:
            report = []

        for item in report:
            with st.container(border=True):
                st.markdown(f"**{item['topic']} — {item['count']} negative feedback items**")
                st.write(item["headline"])
                st.caption("Evidence · expand a review to see its original feedback")
                for evidence in item["evidence"]:
                    with st.expander(f"View original feedback: {evidence['id']}"):
                        original = data.loc[data["id"].eq(evidence["id"])]
                        if not original.empty:
                            review = original.iloc[0]
                            st.write(review["raw_text"])
                            st.caption(f"Topic: {review['topic']}")
                            st.caption(f"Sentiment: {review['sentiment']}")
                        else:
                            st.warning("Original feedback not found.")
                    st.write(f"“{evidence['quote']}”")
                    if evidence.get("source_type") == SUPPORT:
                        st.caption(
                            f"Support ticket: {evidence.get('ticket_id') or 'not supplied'} "
                            f"· Channel: {evidence.get('channel') or 'not supplied'}"
                        )

with tab_emerging:
    st.subheader("Candidate new issues")
    st.caption("Discover patterns in recurring customer complaints that may need further investigation.")
    cutoff = st.slider("Low-confidence threshold", 0.25, 0.90, 0.55, 0.05)

    clusters = possible_new_issues(
        data,
        threshold=cutoff,
        min_reports=3,
    )
    if not clusters:
        st.info("No repeated, sufficiently similar negative candidate groups at this setting.")
    for cluster in clusters:
        with st.expander(f"Candidate cluster {cluster['group']} · {cluster['count']} reports"):
            st.write(f"**Issue status:** {cluster['status']}")
            if cluster.get("matched_known_issue"):
                st.caption(f"Matched known issue: {cluster['matched_known_issue']}")
            st.caption(cluster["reason"])
            for example in cluster["examples"]:
                link = url_quote(str(example['id']), safe='')
                st.markdown(f"**[{example['id']}](?review_id={link})**")
                st.write(example["raw_text"])

with tab_evaluation:
    st.subheader("Held-out model comparison")
    path = ROOT / "outputs" / "evaluation_results.json"
    if not path.exists():
        st.warning("Official results pending: hand-check at least 100 rows in data/evaluation_to_verify.csv, mark human_verified=yes, and run python -m src.evaluate.")
        st.code("python -m pip install -r requirements-embeddings.txt\npython -m src.evaluate", language="powershell")
    else:
        result = json.loads(path.read_text(encoding="utf-8"))
        entries = [{"task / model": name, "accuracy": details["accuracy"], "macro F1": details["macro_f1"],
                    "fit seconds": details.get("training_seconds"), "predict seconds": details.get("prediction_seconds")}
                   for name, details in result["scores"].items()]
        st.caption(f"Hand-verified held-out samples: {result['verified_rows']} · Training rows: {result['training_rows']}. All methods evaluated using the same IDs.")
        st.dataframe(pd.DataFrame(entries), hide_index=True, width="stretch")
        chosen = st.selectbox("View confusion matrix", list(result["scores"]))
        matrix = result["scores"][chosen]
        st.plotly_chart(px.imshow(matrix["confusion_matrix"], x=matrix["labels"], y=matrix["labels"], text_auto=True, labels={"x":"Predicted","y":"Actual"}), width="stretch")
        st.caption("For five analysed errors, complete the why_the_model_failed column in outputs/mistakes_to_analyse.csv.")
