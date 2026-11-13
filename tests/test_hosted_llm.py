from unittest.mock import patch, Mock
import pandas as pd
import pytest
from src.core import clean_reviews
from src.summaries import issue_groups, cloud_llm_summary


def test_hosted_api_request_and_evidence_validation():
    data, _ = clean_reviews(pd.DataFrame({"id": ["R1"], "text": ["Delivery was late."]}))
    data["topic"] = "Delivery"
    data["sentiment"] = "negative"
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": '{"items":[{"topic":"Delivery","headline":"Delivery was late.","evidence_ids":["R1"]}]}'}}]}
    with patch("src.summaries.requests.post", return_value=response) as post:
        output, accepted = cloud_llm_summary(issue_groups(data), api_key="TEST_KEY")
    assert accepted == 1
    assert output[0]["source"] == "LLM + validated evidence"
    args, kwargs = post.call_args
    assert args[0] == "https://api.groq.com/openai/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer TEST_KEY"
    assert kwargs["json"]["response_format"] == {"type": "json_object"}
    assert "TEST_KEY" not in str(output)


def test_hosted_api_rejects_invented_evidence_ids():
    data, _ = clean_reviews(pd.DataFrame({"id": ["R1"], "text": ["Delivery was late."]}))
    data["topic"] = "Delivery"
    data["sentiment"] = "negative"
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": '{"items":[{"topic":"Delivery","headline":"Delivery was late.","evidence_ids":["FAKE"]}]}'}}]}
    with patch("src.summaries.requests.post", return_value=response):
        output, accepted = cloud_llm_summary(issue_groups(data), api_key="TEST_KEY")
    assert accepted == 0
    assert output[0]["source"] == "extractive"


def test_missing_key_blocks_api_call():
    with patch("src.summaries.requests.post") as post:
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            cloud_llm_summary([{"topic":"Delivery", "count":1, "evidence":[{"id":"R1", "quote":"Late delivery"}]}], api_key="")
        post.assert_not_called()
