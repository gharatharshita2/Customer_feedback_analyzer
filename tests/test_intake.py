import pandas as pd
import pytest
from src.intake import REVIEW, SUPPORT, prepare_source, combine_sources
from src.summaries import issue_groups


def test_two_sources_and_provenance_and_agent_reply_skipped():
    reviews = pd.DataFrame({"id": ["1"], "date": ["2026-10-10"], "review": ["This product was good."]})
    messages = pd.DataFrame({
        "message_id": ["1", "2", "3"],
        "ticket_id": ["T1", "T1", "T2"],
        "sender_role": ["customer", "agent", "customer"],
        "created_at": ["2026-10-11"]*3,
        "channel": ["chat"]*3,
        "message": ["My parcel arrived late.", "Sorry for the delay.", "My parcel arrived late."]
    })
    data, stats = combine_sources([(reviews, REVIEW), (messages, SUPPORT)])
    assert len(data) == 3  # repeated complaint in another ticket is counted
    assert stats["raw"] == 4
    assert stats["skipped_non_customer"] == 1
    assert data.id.nunique() == 3
    assert set(data.source_type) == {REVIEW, SUPPORT}
    assert data.loc[data.source_type.eq(SUPPORT), "ticket_id"].tolist() == ["T1", "T2"]
    assert stats["duplicates"] == 0


def test_duplicate_within_same_support_ticket_removed():
    messages = pd.DataFrame({"ticket_id": ["T1", "T1", "T2"], "message": ["Please refund my money!", " please   refund my MONEY! ", "Please refund my money!"]})
    data, stats = prepare_source(messages, SUPPORT)
    assert len(data) == 2
    assert stats["duplicates"] == 1
    assert stats["usable"] == 2


def test_support_message_only_with_flexible_columns_and_bad_csv():
    messages = pd.DataFrame({"Message Body": ["The website keeps crashing."], "Case ID": ["C-9"], "Sent At": ["2026-10-20"], "Sender Role": ["customer"]})
    data, stats = combine_sources([(messages, SUPPORT)])
    assert stats["usable"] == 1
    assert data.iloc[0].ticket_id == "C-9"
    assert str(data.iloc[0].date.date()) == "2026-10-20"
    with pytest.raises(ValueError, match="needs a feedback column"):
        prepare_source(pd.DataFrame({"ticket_id": ["T1"]}), SUPPORT)
    with pytest.raises(ValueError, match="no customer messages"):
        prepare_source(pd.DataFrame({"sender_role": ["bot"], "message": ["We got your ticket."]}), SUPPORT)


def test_support_quote_is_verbatim_and_is_evidence():
    messages = pd.DataFrame({"message_id": ["M10"], "ticket_id": ["T123"], "message": ["My refund is still missing!"], "channel": ["email"]})
    data, _ = combine_sources([(messages, SUPPORT)])
    data["topic"] = "Refunds"
    data["sentiment"] = "negative"
    groups = issue_groups(data)
    assert groups[0]["evidence"][0]["quote"] == "My refund is still missing!"
    assert groups[0]["evidence"][0]["ticket_id"] == "T123"
    assert groups[0]["evidence"][0]["id"] == "SUP:M10"


def test_blank_support_message_is_cleanly_rejected():
    with pytest.raises(ValueError, match="No usable reviews"):
        prepare_source(pd.DataFrame({"message": ["  "]}), SUPPORT)


def test_combined_summary_cites_both_sources():
    reviews = pd.DataFrame({"id": ["1", "2"], "text": ["My parcel arrived late.", "The courier never showed up."]})
    messages = pd.DataFrame({"message_id": ["10"], "ticket_id": ["T10"], "message": ["Delivery is missing again."], "channel": ["chat"]})
    data, _ = combine_sources([(reviews, REVIEW), (messages, SUPPORT)])
    data["topic"] = "Delivery"
    data["sentiment"] = "negative"
    groups = issue_groups(data)
    evidence = groups[0]["evidence"]
    assert {e["source_type"] for e in evidence} == {REVIEW, SUPPORT}
    assert sum(1 for e in evidence if e["id"].startswith("SUP:")) == 1
