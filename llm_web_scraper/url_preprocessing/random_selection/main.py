import random
import logging
from typing import List

logger = logging.getLogger(__name__)


async def random_selection(
    base_url: str,
    num_results: int = 10,
) -> List[str]:
    from llm_web_scraper.url_preprocessing.crawl_search import get_crawl_search_results

    crawled_urls, _ = await get_crawl_search_results(
        user_intent="",
        root_url=base_url,
        num_results=num_results,
        search_keywords="",
    )

    if not crawled_urls:
        return []

    num_to_select = min(num_results, len(crawled_urls))
    return random.sample(crawled_urls, num_to_select)


async def get_random_selection_results(
    user_intent: str,
    context_url: str,
    num_results: int,
) -> List[str]:
    return await random_selection(context_url, num_results)
