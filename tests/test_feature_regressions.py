"""Focused regression tests for interview feedback; no live external requests."""
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from src.aspects import explain_review
from src.core import load_csv
from src.image_feedback import describe_image
from src.intake import ingest_feedback_csv
from src.language import detect_language, translate_for_analysis
from src.models import possible_new_issues
from src.summaries import _post_with_retry, issue_groups


def _data(texts, sentiment='negative'):
    return pd.DataFrame({
        'id': [f'R{i}' for i in range(len(texts))],
        'text': texts,
        'raw_text': texts,
        'topic': ['Other'] * len(texts),
        'confidence': [0.25] * len(texts),
        'sentiment': [sentiment] * len(texts),
    })


def test_mixed_returns_both_aspects_and_primary_negative():
    d = explain_review('The product quality is excellent, but delivery was late.', 'Product Quality', 'positive')
    assert d['mixed'] is True
    assert (d['topic'], d['sentiment']) == ('Delivery', 'negative')
    assert ('Product Quality', 'positive') in {(a['topic'], a['sentiment']) for a in d['aspects']}
    assert ('Delivery', 'negative') in {(a['topic'], a['sentiment']) for a in d['aspects']}


def test_another_mixed_case_with_failed_charger():
    d = explain_review('Shipping was quick, but the charger stopped working.', 'Delivery', 'positive')
    assert d['mixed']
    assert ('Product Quality','negative') in {(a['topic'], a['sentiment']) for a in d['aspects']}


def test_positive_feedback_is_not_negative():
    d = explain_review('Excellent product quality, very happy.', 'Product Quality', 'negative')
    assert d['sentiment'] == 'positive'


def test_known_issue_remains_visible_and_has_status():
    texts = [f'My loyalty points disappeared from my account again case {i}' for i in range(8)]
    output = possible_new_issues(_data(texts), known_issues=('loyalty points',), min_reports=3)
    assert output
    assert output[0]['status'] == 'Known recurring issue'
    assert output[0]['matched_known_issue'] == 'loyalty points'
    assert output[0]['count'] == 8


def test_repeated_unmatched_complaint_is_candidate_not_confirmed():
    texts = [f'My charger overheated and the plastic melted issue {i}' for i in range(7)]
    output = possible_new_issues(_data(texts), known_issues=('loyalty points',))
    assert output
    assert output[0]['status'] == 'Possible new issue - needs review'


def test_approved_variation_can_be_excluded():
    texts = [f'Minor color variation due to lighting on item {i}' for i in range(7)]
    assert not possible_new_issues(_data(texts), known_variations=('minor color variation due to lighting',))


def test_positive_or_single_negative_not_emerging_issue():
    texts = ['points disappeared today', 'points disappeared again', 'points disappeared yesterday']
    assert not possible_new_issues(_data(texts, sentiment='positive'))
    assert not possible_new_issues(_data([texts[0]]))


def test_429_retry_then_success_without_real_network():
    first = Mock(status_code=429, headers={'Retry-After':'0'})
    second = Mock(status_code=200, headers={})
    post = Mock(side_effect=[first, second])
    with patch('src.summaries.requests.post', post), patch('src.summaries.time.sleep'):
        result = _post_with_retry('https://example.invalid', timeout=1)
    assert result is second
    assert post.call_count == 2


def test_timeout_is_bounded():
    post = Mock(side_effect=requests.Timeout('simulated timeout'))
    with patch('src.summaries.requests.post', post), patch('src.summaries.time.sleep'):
        with pytest.raises(requests.Timeout):
            _post_with_retry('https://example.invalid')
    assert post.call_count == 2


def test_auth_error_does_not_retry():
    denied = Mock(status_code=401, headers={})
    denied.raise_for_status.side_effect = requests.HTTPError('401')
    post = Mock(return_value=denied)
    with patch('src.summaries.requests.post', post):
        with pytest.raises(requests.HTTPError):
            _post_with_retry('https://example.invalid')
    assert post.call_count == 1


def test_obvious_language_flags():
    assert detect_language('मेरा ऑर्डर अभी तक नहीं आया') == 'Hindi/Devanagari (possible)'
    assert detect_language('Mera order abhi tak deliver nahi hua') == 'Hinglish (possible)'
    assert detect_language('The delivery was late') == 'English/undetermined'


def test_translation_preserves_ids_and_rejects_forged_result():
    rows = pd.DataFrame({'id':['R1'], 'text':['Mera order nahi aya']})
    response = Mock()
    response.json.return_value = {'choices':[{'message':{'content':'{"items":[{"id":"R1","english":"My order did not arrive"}]}'}}]}
    with patch('src.language._post_with_retry', return_value=response):
        translation = translate_for_analysis(rows, api_key='TEST')
    assert translation == {'R1':'My order did not arrive'}
    response.json.return_value = {'choices':[{'message':{'content':'{"items":[{"id":"FAKE","english":"An invention"}]}'}}]}
    with patch('src.language._post_with_retry', return_value=response):
        with pytest.raises(RuntimeError):
            translate_for_analysis(rows, api_key='TEST')


def test_invalid_image_is_rejected_without_api_request():
    with patch('src.image_feedback._post_with_retry') as post:
        with pytest.raises(ValueError, match='Invalid PNG'):
            describe_image(b'invalid', 'image/png', api_key='TEST')
        post.assert_not_called()


def test_image_description_uses_mocked_vision_response():
    reply = Mock()
    reply.json.return_value = {'choices':[{'message':{'content':'A scratched red case is visible.'}}]}
    with patch('src.image_feedback._post_with_retry', return_value=reply) as post:
        caption = describe_image(b'\x89PNG\r\n\x1a\n' + b'x'*8, 'image/png', api_key='TEST')
    assert caption == 'A scratched red case is visible.'
    payload = post.call_args.kwargs['json']
    assert payload['messages'][1]['content'][1]['type'] == 'image_url'


def test_image_only_is_not_quoted_as_customer_feedback():
    df = pd.DataFrame({'id':['I1'], 'raw_text':['AI described a scratched case'],
                       'topic':['Product Quality'], 'sentiment':['negative'], 'is_image_caption':[True]})
    assert issue_groups(df) == []


def test_large_uploaded_csv_intake_smoke():
    path = Path(__file__).resolve().parents[1] / 'data' / 'large_test_feedback.csv'
    cleaned, stats = ingest_feedback_csv(load_csv(path))
    assert stats['raw'] == 2040
    assert stats['usable'] >= 1500
    assert stats['duplicates'] >= 1
    assert stats['invalid'] >= 1
    assert cleaned['id'].is_unique
    assert {'Review','Support message'} == set(cleaned['source_type'])
