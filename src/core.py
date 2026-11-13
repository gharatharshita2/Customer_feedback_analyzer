"""CSV validation, cleaning and traceable source records."""
from __future__ import annotations
import re
import pandas as pd

TOPICS = ["Delivery", "Product Quality", "Pricing", "Refunds", "Customer Support", "Packaging", "Website/App", "Payments", "Returns", "Other"]
SENTIMENTS = ["negative", "neutral", "positive"]
TEXT_COLUMNS = ("text", "review", "message", "feedback", "review_text")
URGENT_PATTERNS = [
    r"\b(?:unsafe|injur(?:y|ed|ies)|fire hazard|caught fire|electrocution|electric shock)\b",
    r"\b(?:legal action|lawsuit|police complaint|fraud|data breach|stolen card details)\b",
]

def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()

def load_csv(source) -> pd.DataFrame:
    try:
        return pd.read_csv(source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Could not read CSV. Save a valid UTF-8 CSV with a 'text' column. Details: {exc}") from exc

def clean_reviews(raw: pd.DataFrame, *, dedupe_with_ticket: bool = False) -> tuple[pd.DataFrame, dict]:
    if raw is None or raw.empty:
        raise ValueError("The CSV is empty. Add reviews and a 'text' (or 'review') column.")
    df = raw.copy()
    df.columns = [str(name).strip().lower() for name in df.columns]
    field = next((name for name in TEXT_COLUMNS if name in df.columns), None)
    if field is None:
        raise ValueError("Missing review column. Include one of: text, review, message, feedback, review_text.")
    df["raw_text"] = df[field].where(df[field].map(lambda x: isinstance(x, str)), "")
    df["text"] = df["raw_text"].map(clean_text)
    incoming = len(df)
    df = df[df["text"].str.len() >= 3].copy()
    invalid = incoming - len(df)
    # Reviews are deduplicated by cleaned text. For messages, keep identical
    # complaints from different tickets; repeated complaints are real evidence.
    key = df["text"].str.casefold()
    if dedupe_with_ticket and "ticket_id" in df.columns:
        ticket = df["ticket_id"].fillna("").astype(str).str.strip().str.casefold()
        key = ticket + "\x1f" + key
    duplicates = int(key.duplicated().sum())
    df = df.loc[~key.duplicated()].reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable reviews remain after removing empty or duplicate text.")
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    if "id" not in df:
        df["id"] = ""
    df["id"] = df["id"].fillna("").astype(str).str.strip()
    # Duplicate/blank IDs make citations ambiguous; replace only these, not legitimate IDs.
    used = set()
    for i, identifier in enumerate(df["id"]):
        if not identifier or identifier in used:
            identifier = f"ROW-{i+1:05d}"
            while identifier in used:
                identifier += "-x"
        df.at[i, "id"] = identifier
        used.add(identifier)
    df["urgent"] = df["text"].str.contains("|".join(URGENT_PATTERNS), regex=True, case=False)
    stats = {"raw": incoming, "usable": len(df), "invalid": invalid, "duplicates": duplicates, "missing_dates": int(df["date"].isna().sum())}
    return df, stats

def week_starts(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("W-SUN").dt.start_time
