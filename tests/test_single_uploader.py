"""CSV intake behind the one Upload Feedback File control."""
import pandas as pd
import pytest

from src.intake import ingest_feedback_csv, REVIEW, SUPPORT
from src.core import load_csv


def test_reviews_csv_is_detected_automatically():
    raw = pd.DataFrame({"review_id": ["R-1", "R-2"], "date": ["2026-10-01"]*2, "review": ["Wonderful quality!", "The parcel arrived far too late."]})
    df, stats = ingest_feedback_csv(raw)
    assert len(df) == 2 and stats["raw"] == 2
    assert set(df["source_type"]) == {REVIEW}
    assert df["id"].tolist() == ["REV:R-1", "REV:R-2"]


def test_support_csv_is_detected_automatically():
    raw = pd.DataFrame({"message_id": ["M-1", "M-2"], "ticket_id": ["T-1"]*2, "sender_role": ["customer", "agent"], "message": ["I am still waiting for a refund.", "We are checking."]})
    df, stats = ingest_feedback_csv(raw)
    assert stats["raw"] == 2 and stats["usable"] == 1
    assert stats["skipped_non_customer"] == 1
    assert df.iloc[0]["id"] == "SUP:M-1"
    assert df.iloc[0]["ticket_id"] == "T-1"


def test_mixed_file_with_explicit_source_type():
    raw = pd.DataFrame({
        "id": ["R1", "M1", "M2", "R2"],
        "source_type": ["Review", "Support message", "support", "review"],
        "ticket_id": ["", "T1", "T1", ""],
        "sender_role": ["", "customer", "agent", ""],
        "text": ["Excellent jacket", "Refund has not arrived", "We will check", "Late delivery"],
    })
    df, stats = ingest_feedback_csv(raw)
    assert stats["raw"] == 4 and stats["usable"] == 3
    assert stats["skipped_non_customer"] == 1
    assert set(df["id"]) == {"REV:R1", "REV:R2", "SUP:M1"}


def test_sample_file_included_in_package():
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "data" / "sample_feedback_upload.csv"
    df, stats = ingest_feedback_csv(load_csv(src))
    assert stats["raw"] == 10
    assert stats["skipped_non_customer"] == 1
    assert stats["usable"] == 9
    assert set(df["source_type"]) == {REVIEW, SUPPORT}


def test_invalid_csv_and_source_are_explained():
    with pytest.raises(ValueError, match="empty"):
        ingest_feedback_csv(pd.DataFrame())
    with pytest.raises(ValueError, match="Invalid source_type"):
        ingest_feedback_csv(pd.DataFrame({"source_type": ["unknown"], "text": ["Nice"]}))
    with pytest.raises(ValueError, match="source_type column"):
        ingest_feedback_csv(pd.DataFrame({"message": ["Broken item"], "review": ["Late delivery"]}))
    with pytest.raises(ValueError, match="feedback column"):
        ingest_feedback_csv(pd.DataFrame({"some_column": ["x"]}))
