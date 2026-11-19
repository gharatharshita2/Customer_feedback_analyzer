"""Optional image-attached feedback helper, separate from CSV ground truth.

A vision model can describe visible evidence. It cannot establish what a customer
meant or provide a verbatim customer quote. A human must confirm the description
before it is used for a single-feedback classification demo.
"""
from __future__ import annotations
import base64
import os
import requests
from .summaries import _post_with_retry

MAX_IMAGE_BYTES = 4 * 1024 * 1024
DEFAULT_VISION_MODEL = 'qwen/qwen3.8-27b'


def describe_image(image_bytes: bytes, mime: str, *, api_key: str, model: str | None = None) -> str:
    if mime not in ('image/png', 'image/jpeg'):
        raise ValueError('Only PNG and JPEG images are supported.')
    if not 0 < len(image_bytes) <= MAX_IMAGE_BYTES:
        raise ValueError('Image must be smaller than 4 MB.')
    if mime == 'image/png' and not image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Invalid PNG file.')
    if mime == 'image/jpeg' and not image_bytes.startswith(b'\xff\xd8\xff'):
        raise ValueError('Invalid JPEG file.')
    if not api_key:
        raise RuntimeError('Groq API key is needed for image descriptions.')
    selected_model = model or os.environ.get('GROQ_VISION_MODEL', DEFAULT_VISION_MODEL)
    image_url = f'data:{mime};base64,{base64.b64encode(image_bytes).decode("ascii")}'
    try:
        response = _post_with_retry(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization':f'Bearer {api_key}', 'Content-Type':'application/json'},
            json={'model': selected_model,
                  'messages':[
                      {'role':'system','content':'Describe only visible objects, packaging and defects. Do not infer customer intent, blame, sentiment, dates, or causes.'},
                      {'role':'user','content':[
                          {'type':'text','text':'In one or two sentences, describe only observable evidence in this image. No unsupported assertions.'},
                          {'type':'image_url','image_url':{'url':image_url}},
                      ]},
                  ], 'temperature':0, 'max_completion_tokens':180},
            timeout=40,
        )
        text = response.json()['choices'][0]['message']['content']
        if not isinstance(text,str) or not 5 <= len(text.strip()) <= 1500:
            raise ValueError('Invalid vision response')
        return text.strip()
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError('Image description unavailable. Please enter a manual description.') from exc
