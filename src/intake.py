"""Accept a single CSV of reviews, support messages, or both.

This is an ingestion adapter, not a separate model: both sources flow through
exactly the same cleaning, classification, trend and evidence pipeline.
"""
from __future__ import annotations
import re
import pandas as pd

from .core import clean_reviews

REVIEW = "Review"
SUPPORT = "Support message"

REVIEW_TEXT_FIELDS = ("text", "review_text", "review", "feedback", "content", "body")
SUPPORT_TEXT_FIELDS = ("message", "message_text", "message_body", "body", "content", "text", "feedback")
DATE_FIELDS = ("date", "created_at", "timestamp", "sent_at", "submitted_at", "review_date")
TICKET_FIELDS = ("ticket_id", "support_ticket_id", "conversation_id", "case_id", "thread_id")
CHANNEL_FIELDS = ("channel", "support_channel", "contact_channel", "platform")
ROLE_FIELDS = ("sender_role", "author_role", "sender_type", "role", "speaker")
NON_CUSTOMER_ROLES = {
    "agent", "support_agent", "support_staff", "customer_support_agent",
    "staff", "team", "employee", "representative", "operator", "admin",
    "system", "bot", "chatbot", "assistant", "automation", "merchant",
}


def _first(columns, choices):
    return next((name for name in choices if name in columns), None)


def prepare_source(raw: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict]:
    """Standardize, validate, dedupe and label one CSV source.

    Agent and bot replies must not inflate customer sentiment/counts. Missing or
    unrecognized roles are retained (upload only public or synthetic customer data).
    """
    if source not in (REVIEW, SUPPORT):
        raise ValueError("Unknown feedback source")
    if raw is None or raw.empty:
        raise ValueError(f"{source} CSV is empty.")

    df = raw.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError(f"{source} CSV contains duplicate column names")
    field = _first(df.columns, SUPPORT_TEXT_FIELDS if source == SUPPORT else REVIEW_TEXT_FIELDS)
    if field is None:
        expected = "message, body, content or text" if source == SUPPORT else "text, review or feedback"
        raise ValueError(f"{source} CSV needs a feedback column ({expected}).")

    total = len(df)
    skipped = 0
    if source == SUPPORT:
        role_field = _first(df.columns, ROLE_FIELDS)
        if role_field:
            role = df[role_field].fillna("").astype(str).str.casefold().str.strip()
            role = role.str.replace(r"[\s-]+", "_", regex=True)
            exclude = role.isin(NON_CUSTOMER_ROLES)
            skipped = int(exclude.sum())
            df = df.loc[~exclude].copy()
        if df.empty:
            raise ValueError("The support CSV has no customer messages after excluding agent/bot replies.")

    df["text"] = df[field]
    # A message_id identifies a single message; a ticket_id identifies its thread.
    id_field = _first(df.columns, ("message_id", "id", "event_id") if source == SUPPORT else ("id", "review_id", "feedback_id"))
    df["id"] = df[id_field] if id_field else ""
    date_field = _first(df.columns, DATE_FIELDS)
    df["date"] = df[date_field] if date_field else pd.NaT
    ticket_field = _first(df.columns, TICKET_FIELDS)
    channel_field = _first(df.columns, CHANNEL_FIELDS)
    df["ticket_id"] = df[ticket_field].fillna("").astype(str).str.strip() if ticket_field else ""
    df["channel"] = df[channel_field].fillna("").astype(str).str.strip() if channel_field else ""
    cleaned, stats = clean_reviews(df, dedupe_with_ticket=(source == SUPPORT))
    cleaned["source_type"] = source
    cleaned["source_id"] = cleaned["id"]
    cleaned["id"] = cleaned["id"].map(lambda x: ("SUP:" if source == SUPPORT else "REV:") + x)
    stats["raw"] = total
    stats["skipped_non_customer"] = skipped
    return cleaned, stats


def combine_sources(inputs: list[tuple[pd.DataFrame, str]]) -> tuple[pd.DataFrame, dict]:
    if not inputs:
        raise ValueError("Upload customer reviews or support messages to analyse feedback.")
    frames = []
    totals = {"raw": 0, "usable": 0, "invalid": 0, "duplicates": 0,
              "missing_dates": 0, "skipped_non_customer": 0}
    for raw, source in inputs:
        frame, stats = prepare_source(raw, source)
        frames.append(frame)
        for key in totals:
            totals[key] += int(stats.get(key, 0))
    result = pd.concat(frames, ignore_index=True)
    if result["id"].duplicated().any():
        raise ValueError("Source IDs must be unique. Check review/message identifiers.")
    return result, totals


# The UI exposes one upload control; this function selects the source adapter.
# Mixed exports can supply a source_type column to mark records row by row.
SOURCE_COLUMN_NAMES = ("source_type", "feedback_type", "record_type")
REVIEW_SOURCE_NAMES = {"review", "reviews", "customer_review", "customer_reviews"}
SUPPORT_SOURCE_NAMES = {
    "support", "support_message", "support_messages", "customer_support",
    "customer_support_message", "ticket", "ticket_message", "message", "messages",
}


def ingest_feedback_csv(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ingest one uploaded CSV, auto-detecting review or support data.

    If `source_type` / `feedback_type` / `record_type` is present, split rows by
    their explicit source. For one-source exports, detect based on the column
    names (message/ticket metadata => support; otherwise => review).
    """
    if raw is None or raw.empty:
        raise ValueError("Feedback CSV is empty. Upload a CSV containing customer feedback.")
    df = raw.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("Feedback CSV contains duplicate column names.")

    source_column = _first(df.columns, SOURCE_COLUMN_NAMES)
    if source_column:
        values = df[source_column].fillna("").astype(str).str.casefold().str.strip()
        values = values.str.replace(r"[\s/-]+", "_", regex=True)
        values = values.map(lambda name: REVIEW if name in REVIEW_SOURCE_NAMES else SUPPORT if name in SUPPORT_SOURCE_NAMES else "")
        if (values == "").any():
            bad_count = int((values == "").sum())
            raise ValueError(f"Invalid source_type on {bad_count} row(s). Use 'Review' or 'Support message'.")
        return combine_sources([(df.loc[values == source].copy(), source) for source in (REVIEW, SUPPORT) if (values == source).any()])

    columns = set(df.columns)
    # Ticket/message headers are strong support signals, even when the body is
    # stored under a generic 'text' or 'feedback' column.
    support_markers = set(TICKET_FIELDS) | set(ROLE_FIELDS) | {"message_id", "event_id", "message", "message_body", "message_text"}
    review_markers = {"review_id", "review_text", "review"}
    if columns & support_markers and columns & review_markers:
        raise ValueError("CSV contains both review and support-specific columns. Add a source_type column ('Review' or 'Support message') to distinguish rows.")
    source = SUPPORT if columns & support_markers else REVIEW
    return combine_sources([(df, source)])
