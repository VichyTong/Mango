import re
import requests
from typing import List, Tuple, Any
from bs4 import BeautifulSoup, Tag
from rank_bm25 import BM25Okapi
from crawl4ai.utils import clean_tokens
from Stemmer import Stemmer
import jieba

class BM25ContentFilter:
    def __init__(
        self,
        bm25_threshold: float = 1.0,
        language: str = "english",
        use_stemming: bool = True,
    ):
        self.bm25_threshold = bm25_threshold
        self.use_stemming = use_stemming
        self.language = language
        self.priority_tags = {
            "h1": 5.0, "h2": 4.0, "h3": 3.0, "title": 4.0,
            "strong": 2.0, "b": 1.5, "em": 1.5,
            "blockquote": 2.0, "code": 2.0, "pre": 1.5, "th": 1.5
        }

        if use_stemming and language.lower() not in ['chinese', 'japanese', 'korean']:
            try:
                self.stemmer = Stemmer(language)
            except KeyError:
                self.stemmer = None
        else:
            self.stemmer = None

    def fetch_url(self, url: str, timeout: int = 10) -> str:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return ""

    def _tokenize(self, text: str, stemmer: Any = None) -> List[str]:
        if self.language.lower() in ['chinese', 'zh', 'zh-cn', 'zh-tw']:
            tokens = list(jieba.cut(text.lower()))
            tokens = [t.strip() for t in tokens if t.strip() and not re.match(r'^[\s\W]+$', t)]
        else:
            text = re.sub(r'[^\w\s]|_', ' ', text.lower())
            tokens = text.split()
            if stemmer:
                tokens = stemmer.stemWords(tokens)
        return clean_tokens(tokens)

    def calculate_bm25_scores_for_urls(
        self,
        urls: List[str],
        query: str,
        htmls: List[str] = None
    ) -> List[float]:
        
        tokenized_query = self._tokenize(query, self.stemmer)
        
        tokenized_corpus = []
        valid_urls_mask = [] 

        for url in urls:
            if htmls:
                html = htmls[urls.index(url)]
            else:
                html = self.fetch_url(url)
            if not html:
                tokenized_corpus.append([])
                continue
            soup = BeautifulSoup(html, "lxml")
            if soup.body:
                text = soup.body.get_text(" ", strip=True)
            else:
                text = soup.get_text(" ", strip=True)
                
            tokens = self._tokenize(text, self.stemmer)
            tokenized_corpus.append(tokens)

        if not tokenized_corpus or not any(tokenized_corpus):
             return [0.0] * len(urls)

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
        
        return scores.tolist()

    def extract_page_query(self, soup: BeautifulSoup, body: Tag) -> str:
        if self.user_query:
            return self.user_query
        
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        
        h1 = body.find('h1')
        if h1:
            return h1.get_text(strip=True)
            
        return ""

    def extract_text_chunks(self, body: Tag, min_word_threshold: int = None) -> List[Tuple[int, str, str, Tag]]:
        candidates = []
        index = 0
        min_words = min_word_threshold if min_word_threshold else 5

        tags_to_check = ['p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'pre']
        
        for element in body.find_all(tags_to_check):
            text = element.get_text(" ", strip=True)
            word_count = len(text.split())
            
            if word_count >= min_words:
                candidates.append((index, text, element.name, element))
                index += 1
                
        return candidates

    def clean_element(self, tag: Tag) -> str:
        return tag.get_text(" ", strip=True)

    def filter_content(self, source: str, is_url: bool = False, min_word_threshold: int = None) -> List[str]:
        html = ""
        if is_url:
            html = self.fetch_url(source)
        else:
            html = source

        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")

        if not soup.body:
            soup = BeautifulSoup(f"<body>{html}</body>", "lxml")
        body = soup.find("body")

        query = self.extract_page_query(soup, body)

        if not query:
            return []

        candidates = self.extract_text_chunks(body, min_word_threshold)

        if not candidates:
            return []

        tokenized_corpus = [
            self._tokenize(chunk, self.stemmer) for _, chunk, _, _ in candidates
        ]
        tokenized_query = self._tokenize(query, self.stemmer)

        if not tokenized_query: 
            return [self.clean_element(tag) for _, _, _, tag in candidates]

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        adjusted_candidates = []
        for score, (index, chunk, tag_type, tag) in zip(scores, candidates):
            tag_weight = self.priority_tags.get(tag_type, 1.0)
            adjusted_score = score * tag_weight
            adjusted_candidates.append((adjusted_score, index, chunk, tag))

        selected_candidates = [
            (index, chunk, tag)
            for adjusted_score, index, chunk, tag in adjusted_candidates
            if adjusted_score >= self.bm25_threshold
        ]

        if not selected_candidates:
            return []

        selected_candidates.sort(key=lambda x: x[0])

        return [self.clean_element(tag) for _, _, tag in selected_candidates]