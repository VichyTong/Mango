import re
import logging
from typing import List

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

_model_cache: dict = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        logger.info(f"Loading sentence-transformers model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
    return _model_cache[model_name]


def _fetch_url(url: str, timeout: int = 10) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return ""


def _extract_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    if soup.body:
        text = soup.body.get_text(" ", strip=True)
    else:
        text = soup.get_text(" ", strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:2000]


def calculate_embedding_scores_for_urls(
    urls: List[str],
    query: str,
    model_name: str = "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    htmls: List[str] = None,
) -> List[float]:
    model = _get_model(model_name)

    texts = []
    for i, url in enumerate(urls):
        if htmls is not None and i < len(htmls):
            html = htmls[i]
        else:
            html = _fetch_url(url)
        text = _extract_text(html)
        if not text:
            text = url
        texts.append(text)

    if hasattr(model, 'encode_query') and hasattr(model, 'encode_document'):
        query_embedding = model.encode_query([query])
        doc_embeddings = model.encode_document(texts)
    else:
        query_embedding = model.encode([query])
        doc_embeddings = model.encode(texts)

    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    return similarities.tolist()
