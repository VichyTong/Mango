import re
import time
import logging
from typing import List, Tuple

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

from llm_web_scraper.utils.bm25 import BM25ContentFilter

logger = logging.getLogger(__name__)


browser_cfg = BrowserConfig(
    browser_type="chromium",
    headless=True,
    viewport_width=1280,
    viewport_height=720,
    extra_args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
)


async def get_crawl_search_results(
    user_intent: str,
    root_url: str,
    num_results: int,
    search_keywords: str,
    max_pages: int = 1000,
    scoring_method: str = "bm25",
) -> Tuple[List[str], int]:
    logger.info(f"Starting crawl for {root_url}")
    logger.info(f"Search keywords: {search_keywords}")
    logger.info(f"Max pages: {max_pages}")

    strategy = BFSDeepCrawlStrategy(
        max_depth=3,
        include_external=False,
        max_pages=max_pages
    )

    crawler_cfg = CrawlerRunConfig(
        exclude_external_links=True,
        exclude_social_media_links=True,
        exclude_all_images=True,
        deep_crawl_strategy=strategy
    )

    crawled_urls = []
    crawled_htmls = []

    try:
        crawl_start = time.time()
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            results = await crawler.arun(root_url, config=crawler_cfg)
        crawl_elapsed = time.time() - crawl_start

        logger.info(f"Crawled {len(results)} pages")
        logger.info(f"[TIMING] Crawling: {crawl_elapsed:.3f}s")

        for result in results:
            if result.url and result.url not in crawled_urls:
                crawled_urls.append(result.url)
                crawled_htmls.append(result.html)

        if not crawled_urls:
            logger.warning(f"No URLs crawled for {root_url}, returning root URL")
            return [root_url], 0

        logger.info(f"Starting {scoring_method} ranking for {len(crawled_urls)} URLs")

        try:
            if scoring_method == "embedding":
                from llm_web_scraper.utils.embedding_scorer import calculate_embedding_scores_for_urls
                scores = calculate_embedding_scores_for_urls(
                    urls=crawled_urls,
                    query=user_intent,
                    htmls=crawled_htmls,
                )
            else:
                if re.search(r'[\u4e00-\u9fa5]', user_intent):
                    language = "chinese"
                    use_stemming = False
                else:
                    language = "english"
                    use_stemming = True

                bm25_filter = BM25ContentFilter(
                    use_stemming=use_stemming,
                    language=language
                )
                scores = bm25_filter.calculate_bm25_scores_for_urls(
                    urls=crawled_urls,
                    query=user_intent,
                    htmls=crawled_htmls,
                )

            url_score_pairs = list(zip(crawled_urls, scores))
            url_score_pairs.sort(key=lambda x: x[1], reverse=True)

            top_urls = [url for url, score in url_score_pairs[:num_results]]

            logger.info(f"{scoring_method} ranking complete, returning top {len(top_urls)} URLs")

            return top_urls, 0

        except Exception as e:
            logger.error(f"{scoring_method} ranking failed: {e}, using crawl order as fallback")
            return crawled_urls[:num_results], 0

    except Exception as e:
        logger.error(f"Crawl failed for {root_url}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return [root_url], 0
