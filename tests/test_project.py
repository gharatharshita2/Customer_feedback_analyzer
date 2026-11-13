import json
import pandas as pd
import pytest
from src.core import clean_reviews
from src.aspects import explain_review
from src.summaries import issue_groups, validate_generated, extractive_summary
from src.models import ClassicClassifier, scores, possible_new_issues
from src.evaluate import read_verified

def sample():
    return pd.DataFrame({"id": ["A", "B", "B", "C"], "date": ["2026-10-01"]*4, "text": [
        "Great product, very happy.", "Delivery was late.", " delivery   WAS  late. ", "The support agent threatened legal action."
    ]})

def test_dedup_and_urgent_and_ids():
    df, quality = clean_reviews(sample())
    assert quality["duplicates"] == 1
    assert len(df) == 3
    assert df.id.nunique() == 3
    assert df.urgent.sum() == 1

def test_empty_and_invalid_csv():
    with pytest.raises(ValueError, match="empty"):
        clean_reviews(pd.DataFrame())
    with pytest.raises(ValueError, match="Missing review column"):
        clean_reviews(pd.DataFrame({"rating": [1, 2]}))
    with pytest.raises(ValueError, match="No usable reviews"):
        clean_reviews(pd.DataFrame({"text": ["", " ", None]}))

def test_positive_never_complaint_and_mixed_is_explained():
    positive = explain_review("The product quality is excellent and the material is great.", "Product Quality", "negative")
    assert positive["sentiment"] == "positive"
    assert explain_review("Absolutely fantastic, I love it!", "Other", "negative")["sentiment"] == "positive"
    mixed = explain_review("The product quality is excellent, but delivery was very late.", "Product Quality", "positive")
    assert mixed["mixed"]
    assert (mixed["topic"], mixed["sentiment"]) == ("Delivery", "negative")
    assert "positive Product Quality" in mixed["explanation"]

def test_summary_only_negative_and_quotes_exact():
    df, _ = clean_reviews(sample())
    df["topic"] = ["Product Quality", "Delivery", "Customer Support"]
    df["sentiment"] = ["positive", "negative", "negative"]
    groups = issue_groups(df)
    assert sum(g["count"] for g in groups) == 2
    quotes = [x["quote"] for g in groups for x in g["evidence"]]
    assert "Great product, very happy." not in quotes
    assert "Delivery was late." in quotes
    assert all(g["headline"] for g in extractive_summary(groups))

def test_generated_claim_rejects_fabricated_source_and_claim():
    df, _ = clean_reviews(sample())
    df["topic"], df["sentiment"] = ["Product Quality","Delivery","Customer Support"],["positive","negative","negative"]
    groups = issue_groups(df)
    proposal = {"items": [
        {"topic": "Delivery", "headline": "Customers complained about an explosion.", "evidence_ids": ["B"]},
        {"topic": "Customer Support", "headline": "Customer support threatened legal action.", "evidence_ids": ["FAKE"]},
    ]}
    out, n = validate_generated(proposal, groups)
    assert n == 0
    assert all(g["source"] == "extractive" for g in out)
    assert "explosion" not in str(out)

def test_valid_llm_response_is_citation_locked():
    df, _ = clean_reviews(sample())
    df["topic"], df["sentiment"] = ["Product Quality","Delivery","Customer Support"],["positive","negative","negative"]
    groups = issue_groups(df)
    response = {"items": [{"topic":"Delivery","headline":"Delivery was late.","evidence_ids":["B"]}]}
    out, accepted = validate_generated(response, groups)
    assert accepted == 1
    assert out[0]["evidence"][0]["quote"] == "Delivery was late."

def test_verified_labels_are_really_required(tmp_path):
    p = tmp_path/"labels.csv"
    pd.DataFrame({"id": ["X"], "text":["one"], "topic":["Delivery"], "sentiment":["negative"], "human_verified":["no"]}).to_csv(p,index=False)
    with pytest.raises(ValueError, match="At least 100"):
        read_verified(p)

def test_classic_model_can_train_score():
    texts = ["the parcel arrived late", "courier lost my box", "great and fast delivery", "support reply was rude", "helpdesk is unresponsive", "agent was helpful"]
    y = ["Delivery"]*3 + ["Customer Support"]*3
    model = ClassicClassifier().fit(texts,y)
    result = scores(y,model.predict(texts),["Delivery","Customer Support"])
    assert 0 <= result["macro_f1"] <= 1
    assert len(result["confusion_matrix"]) == 2

def test_clustering_returns_real_source_rows():
    data = pd.DataFrame({"id":["A","B","C","D","E","F"],"topic":["Other"]*6,"confidence":[0.3]*6,
                         "text":["need rewards points", "loyalty points missing", "membership points not showing", "subscription renewal", "monthly membership renewal", "subscription renewal incorrect"],
                         "raw_text":["need rewards points", "loyalty points missing", "membership points not showing", "subscription renewal", "monthly membership renewal", "subscription renewal incorrect"]})
    groups = possible_new_issues(data)
    assert groups
    assert {e['id'] for g in groups for e in g['examples']} <= set(data.id)
