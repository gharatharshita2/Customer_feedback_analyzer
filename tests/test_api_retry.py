
"""Tests for temporary Groq API failures. No real API calls."""

from unittest.mock import Mock

import pytest
import requests

from src.summaries import _post_with_retry


def test_rate_limit_retry(monkeypatch):
    """A 429 rate-limit response should trigger one retry."""

    first = Mock(status_code=429)
    second = Mock(status_code=200)

    fake_post = Mock(side_effect=[first, second])

    monkeypatch.setattr(
        "src.summaries.requests.post", fake_post
    )
    monkeypatch.setattr(
        "src.summaries.time.sleep", lambda seconds: None
    )

    response = _post_with_retry("https://example.invalid")

    assert response.status_code == 200
    assert fake_post.call_count == 2


def test_timeout_retry(monkeypatch):
    """A temporary timeout should trigger one retry."""

    success = Mock(status_code=200)

    fake_post = Mock(
        side_effect=[
            requests.Timeout("Simulated timeout"),
            success,
        ]
    )

    monkeypatch.setattr(
        "src.summaries.requests.post", fake_post
    )
    monkeypatch.setattr(
        "src.summaries.time.sleep", lambda seconds: None
    )

    response = _post_with_retry("https://example.invalid")

    assert response.status_code == 200
    assert fake_post.call_count == 2


def test_repeated_timeout(monkeypatch):
    """Two consecutive timeouts should stop retrying."""

    fake_post = Mock(
        side_effect=requests.Timeout("Simulated timeout")
    )

    monkeypatch.setattr(
        "src.summaries.requests.post", fake_post
    )
    monkeypatch.setattr(
        "src.summaries.time.sleep", lambda seconds: None
    )

    with pytest.raises(requests.Timeout):
        _post_with_retry("https://example.invalid")

    assert fake_post.call_count == 2
