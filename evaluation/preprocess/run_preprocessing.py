import asyncio
import json
import logging
import sys
import os
import re
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llm_web_scraper.url_preprocessing.google_search import get_google_search_results
from llm_web_scraper.url_preprocessing.random_selection import get_random_selection_results
from llm_web_scraper.url_preprocessing.crawl_search import get_crawl_search_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def normalize_model_name_for_path(model_name: str) -> str:
    if model_name.startswith("tensorblock/"):
        return model_name.replace("tensorblock/", "")
    return model_name


def load_google_preprocessing_data(output_dir: str, model_name: str) -> dict:
    normalized_model_name = normalize_model_name_for_path(model_name)
    google_file = os.path.join(
        output_dir,
        "webwalker",
        "preprocess",
        "google",
        normalized_model_name,
        "url_lists.json"
    )

    if not os.path.exists(google_file):
        raise FileNotFoundError(f"Google preprocessing file not found: {google_file}")

    with open(google_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {item['task_index']: item for item in data}


def load_google_preprocessing_data_webvoyager(output_dir: str, model_name: str) -> dict:
    normalized_model_name = normalize_model_name_for_path(model_name)
    google_file = os.path.join(
        output_dir,
        "webvoyager",
        "preprocess",
        "google",
        normalized_model_name,
        "url_lists.json"
    )

    if not os.path.exists(google_file):
        raise FileNotFoundError(f"Google preprocessing file not found: {google_file}")

    with open(google_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {item['task_id']: item for item in data}


async def preprocess_webwalker(method: str, output_dir: str, model_name: str = "gpt-4o-mini", max_urls: int = 10, limit: int = None):
    logger.info(f"Loading WebWalkerQA dataset")
    ds = load_dataset("callanwu/WebWalkerQA", split="main")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    normalized_model_name = normalize_model_name_for_path(model_name)

    if method == "google":
        base_dir = os.path.join(output_dir, "webwalker", "preprocess", "google", normalized_model_name)
    elif method == "crawl":
        base_dir = os.path.join(output_dir, "webwalker", "preprocess", "crawl", normalized_model_name)
    else:
        base_dir = os.path.join(output_dir, "webwalker", "preprocess", "random")

    os.makedirs(base_dir, exist_ok=True)
    output_file = os.path.join(base_dir, "url_lists.json")

    results = []
    completed_indices = set()

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
            for r in existing_results:
                has_error = 'error' in r
                if not has_error:
                    results.append(r)
                    completed_indices.add(r['task_index'])
                else:
                    logger.info(f"Will retry task {r['task_index']} due to error: {r['error']}")
            logger.info(f"Loaded {len(results)} successful results, {len(existing_results) - len(results)} failed results to retry for {method}")

    for idx, item in enumerate(ds):
        if idx in completed_indices:
            logger.info(f"Skipping task {idx + 1}/{len(ds)} (already completed)")
            continue

        root_url = item['root_url']

        logger.info(f"\n{'='*80}")
        logger.info(f"Task {idx + 1}/{len(ds)} - Method: {method}")
        logger.info(f"Root URL: {root_url}")
        logger.info(f"Question: {item['question']}")
        logger.info(f"{'='*80}")

        try:
            if method == "google":
                urls, tokens, search_keywords = await get_google_search_results(item['question'], root_url, max_urls, model_name)
                result = {
                    "task_index": idx,
                    "root_url": root_url,
                    "question": item['question'],
                    "answer": item['answer'],
                    "urls": urls,
                    "url_count": len(urls),
                    "tokens_used": tokens,
                    "search_keywords": search_keywords
                }
            elif method == "crawl":
                google_data = load_google_preprocessing_data(output_dir, model_name)
                task_data = google_data.get(idx)

                if not task_data or 'search_keywords' not in task_data:
                    raise ValueError(f"Missing Google preprocessing data for task {idx}")

                search_keywords = task_data['search_keywords']
                urls, tokens = await get_crawl_search_results(
                    item['question'], root_url, max_urls, search_keywords
                )
                result = {
                    "task_index": idx,
                    "root_url": root_url,
                    "question": item['question'],
                    "answer": item['answer'],
                    "urls": urls,
                    "url_count": len(urls),
                    "tokens_used": tokens,
                    "search_keywords": search_keywords,
                    "crawl_method": "crawl4ai_bm25"
                }
            else:
                urls = await get_random_selection_results(item['question'], root_url, max_urls)
                result = {
                    "task_index": idx,
                    "root_url": root_url,
                    "question": item['question'],
                    "answer": item['answer'],
                    "urls": urls,
                    "url_count": len(urls),
                    "tokens_used": 0
                }

            logger.info(f"Found {len(urls)} URLs")

        except Exception as e:
            logger.error(f"Error processing task {idx}: {e}")
            import traceback
            result = {
                "task_index": idx,
                "root_url": root_url,
                "question": item['question'],
                "answer": item['answer'],
                "urls": [],
                "url_count": 0,
                "tokens_used": 0,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

        results.append(result)
        completed_indices.add(idx)

        sorted_results = sorted(results, key=lambda x: x['task_index'])

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_results, f, indent=4, ensure_ascii=False)

    logger.info(f"\nPreprocessing complete for WebWalkerQA. Results saved to {output_file}")


async def preprocess_webvoyager(method: str, output_dir: str, model_name: str = "gpt-4o-mini", max_urls: int = 10, limit: int = None):
    logger.info(f"Loading WebVoyager dataset")

    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../datasets/webvoyager/WebVoyager_data.jsonl"
    )

    tasks = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    if limit:
        tasks = tasks[:limit]

    normalized_model_name = normalize_model_name_for_path(model_name)

    if method == "google":
        base_dir = os.path.join(output_dir, "webvoyager", "preprocess", "google", normalized_model_name)
    elif method == "crawl":
        base_dir = os.path.join(output_dir, "webvoyager", "preprocess", "crawl", normalized_model_name)
    else:
        base_dir = os.path.join(output_dir, "webvoyager", "preprocess", "random")

    os.makedirs(base_dir, exist_ok=True)
    output_file = os.path.join(base_dir, "url_lists.json")

    results = []
    completed_ids = set()

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
            for r in existing_results:
                has_error = 'error' in r
                is_single_url_chinese = (
                    method == "crawl" and
                    r.get('url_count', 0) == 1 and
                    re.search(r'[\u4e00-\u9fa5]', r.get('question', ''))
                )

                if not has_error and not is_single_url_chinese:
                    results.append(r)
                    completed_ids.add(r['task_id'])
                else:
                    if has_error:
                        logger.info(f"Will retry task {r['task_id']} due to error: {r['error']}")
                    elif is_single_url_chinese:
                        logger.info(f"Will retry task {r['task_id']} - single URL with Chinese question")
            logger.info(f"Loaded {len(results)} successful results, {len(existing_results) - len(results)} failed results to retry for {method}")

    for idx, task in enumerate(tasks):
        task_id = task['id']

        if task_id in completed_ids:
            logger.info(f"Skipping task {idx + 1}/{len(tasks)} (already completed)")
            continue

        root_url = task['web']

        logger.info(f"\n{'='*80}")
        logger.info(f"Task {idx + 1}/{len(tasks)} - Method: {method}")
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Root URL: {root_url}")
        logger.info(f"Question: {task['ques']}")
        logger.info(f"{'='*80}")

        try:
            if method == "google":
                urls, tokens, search_keywords = await get_google_search_results(task['ques'], root_url, max_urls, model_name)
                result = {
                    "task_id": task_id,
                    "root_url": root_url,
                    "web_name": task['web_name'],
                    "question": task['ques'],
                    "urls": urls,
                    "url_count": len(urls),
                    "tokens_used": tokens,
                    "search_keywords": search_keywords
                }
            elif method == "crawl":
                google_data = load_google_preprocessing_data_webvoyager(output_dir, model_name)
                task_data = google_data.get(task_id)

                if not task_data or 'search_keywords' not in task_data:
                    raise ValueError(f"Missing Google preprocessing data for task {task_id}")

                search_keywords = task_data['search_keywords']
                urls, tokens = await get_crawl_search_results(
                    task['ques'], root_url, max_urls, search_keywords
                )
                result = {
                    "task_id": task_id,
                    "root_url": root_url,
                    "web_name": task['web_name'],
                    "question": task['ques'],
                    "urls": urls,
                    "url_count": len(urls),
                    "tokens_used": tokens,
                    "search_keywords": search_keywords,
                    "crawl_method": "crawl4ai_bm25"
                }
            else:
                urls = await get_random_selection_results(task['ques'], root_url, max_urls)
                result = {
                    "task_id": task_id,
                    "root_url": root_url,
                    "web_name": task['web_name'],
                    "question": task['ques'],
                    "urls": urls,
                    "url_count": len(urls),
                    "tokens_used": 0
                }

            logger.info(f"Found {len(urls)} URLs")

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            import traceback
            result = {
                "task_id": task_id,
                "root_url": root_url,
                "web_name": task['web_name'],
                "question": task['ques'],
                "urls": [],
                "url_count": 0,
                "tokens_used": 0,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

        results.append(result)
        completed_ids.add(task_id)

        sorted_results = sorted(results, key=lambda x: x['task_id'])

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_results, f, indent=4, ensure_ascii=False)

    logger.info(f"\nPreprocessing complete for WebVoyager. Results saved to {output_file}")


async def main(
    benchmark: str,
    method: str,
    output_dir: str = "evaluation_results",
    model_name: str = "gpt-4o-mini",
    max_urls: int = 10,
    limit: int = None
):
    if benchmark == "webwalker":
        await preprocess_webwalker(method, output_dir, model_name, max_urls, limit)
    elif benchmark == "webvoyager":
        await preprocess_webvoyager(method, output_dir, model_name, max_urls, limit)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["webwalker", "webvoyager"], required=True)
    parser.add_argument("--method", choices=["google", "random", "crawl"], required=True)
    parser.add_argument("--output", type=str, default="evaluation_results")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-urls", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    asyncio.run(main(
        benchmark=args.benchmark,
        method=args.method,
        output_dir=args.output,
        model_name=args.model,
        max_urls=args.max_urls,
        limit=args.limit
    ))
