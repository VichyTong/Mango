import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from llm_web_scraper.selector.thompson_sampling import thompson_sampling_navigation
from llm_web_scraper.utils.llm import get_litellm_model, create_model_settings

logger = logging.getLogger(__name__)


def normalize_model_name_for_path(model_name: str) -> str:
    if model_name.startswith("tensorblock/"):
        return model_name.replace("tensorblock/", "")
    return model_name


def load_preprocessed_urls(model_name: str, preprocess_dir: str = "evaluation_results/webvoyager/preprocess"):
    normalized_model_name = normalize_model_name_for_path(model_name)

    google_path = os.path.join(preprocess_dir, "google", normalized_model_name, "url_lists.json")
    crawl_path = os.path.join(preprocess_dir, "crawl", normalized_model_name, "url_lists.json")

    with open(google_path, 'r', encoding='utf-8') as f:
        google_data = json.load(f)
    with open(crawl_path, 'r', encoding='utf-8') as f:
        crawl_data = json.load(f)

    return {"google": google_data, "crawl": crawl_data}


def filter_file_urls(urls: list) -> list:
    file_extensions = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.rar', '.tar', '.gz', '.7z',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv',
        '.txt', '.csv', '.xml', '.json'
    }
    return [url for url in urls if not any(url.lower().endswith(ext) for ext in file_extensions)]


def get_urls_for_task(task_id: str, preprocessed_data: dict, root_url: str = None) -> tuple[list, int]:
    google_data = preprocessed_data["google"]
    crawl_data = preprocessed_data["crawl"]

    google_urls, google_tokens = [], 0
    for item in google_data:
        if item['task_id'] == task_id:
            google_urls = item['urls']
            google_tokens = item.get('tokens_used', 0)
            break

    crawl_urls, crawl_tokens = [], 0
    for item in crawl_data:
        if item['task_id'] == task_id:
            crawl_urls = item['urls']
            crawl_tokens = item.get('tokens_used', 0)
            break

    if not google_urls and root_url:
        google_urls = [root_url]

    combined_urls = list(set(crawl_urls) | set(google_urls))
    combined_urls = filter_file_urls(combined_urls)
    combined_tokens = crawl_tokens + google_tokens
    logger.info(f"Task {task_id}: {len(crawl_urls)} crawl + {len(google_urls)} google = {len(combined_urls)} unique")
    return combined_urls, combined_tokens


async def evaluate_one_task(
    question: str,
    root_url: str,
    answer: str,
    task_idx: int,
    task_id: str,
    log_dir: str,
    model_name: str,
    preprocessed_data: dict,
    provider: str = None,
    max_turns_per_navigation: int = 10,
    max_total_navigations: int = 10,
):
    log_file = os.path.join(log_dir, f"{task_idx}.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    logger.info(f"Question: {question}")
    logger.info(f"Root URL: {root_url}")
    logger.info(f"Model Name: {model_name}")

    try:
        urls, url_preprocessing_tokens = get_urls_for_task(task_id, preprocessed_data, root_url)

        if not urls:
            logger.warning(f"No URLs found for {root_url}")
            return {
                "question": question,
                "root_url": root_url,
                "expected_answer": answer,
                "method": "no_memory",
                "success": False,
                "error": "No URLs found",
                "urls_count": 0,
                "url_preprocessing_tokens": url_preprocessing_tokens,
                "navigation_tokens": 0,
                "total_tokens": url_preprocessing_tokens,
            }

        logger.info(f"Found {len(urls)} URLs")

        model = get_litellm_model(model_name=model_name, provider=provider)
        model_settings = create_model_settings(model_name=model_name)

        result = await thompson_sampling_navigation(
            urls=urls,
            user_intent=question,
            model=model,
            model_settings=model_settings,
            max_turns_per_navigation=max_turns_per_navigation,
            max_total_navigations=max_total_navigations,
            navigator_type="mcp",
            use_memory=False,
            metadata={
                "model_name": model_name,
                "method": "no_memory",
                "task_index": str(task_idx),
                "benchmark": "WebVoyager",
            },
        )

        final_output = result.get("final_result", None)
        navigation_tokens = result.get("total_tokens", 0)
        total_tokens = url_preprocessing_tokens + navigation_tokens

        return {
            "question": question,
            "root_url": root_url,
            "expected_answer": answer,
            "method": "no_memory",
            "success": result["final_status"] == "success",
            "total_navigations": result["total_navigations"],
            "urls_count": len(urls),
            "final_output": final_output,
            "url_preprocessing_tokens": url_preprocessing_tokens,
            "navigation_tokens": navigation_tokens,
            "total_tokens": total_tokens,
        }

    except Exception as e:
        error_str = str(e)
        if "Budget limit exceeded" in error_str or "Insufficient balance" in error_str:
            logger.error("Budget limit exceeded - shutting down program")
            root_logger.removeHandler(file_handler)
            file_handler.close()
            sys.exit(1)

        logger.error(f"Error evaluating task: {e}")
        import traceback
        result = {
            "question": question,
            "root_url": root_url,
            "expected_answer": answer,
            "method": "no_memory",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()

    return result


async def main(
    limit: int = None,
    output_dir: str = "evaluation_results",
    model_name: str = "qwen3-32b",
    provider: str = None,
    max_turns_per_navigation: int = 10,
    max_total_navigations: int = 10,
):
    normalized_model_name = normalize_model_name_for_path(model_name)
    base_dir = os.path.join(output_dir, "webvoyager", "rebuttal", normalized_model_name, "no_memory")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    main_log_file = os.path.join(base_dir, "run.log")
    file_handler = logging.FileHandler(main_log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger.info("Loading WebVoyager dataset")
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../datasets/webvoyager/WebVoyager_data.jsonl"
    )

    tasks = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    if limit:
        tasks = tasks[:limit]

    logger.info(f"Evaluating {len(tasks)} tasks")

    preprocessed_data = load_preprocessed_urls(model_name)
    logger.info(f"Loaded preprocessed URLs (google: {len(preprocessed_data['google'])}, crawl: {len(preprocessed_data['crawl'])})")

    output_file = os.path.join(base_dir, "results.json")
    results = []
    completed_indices = set()

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            for r in results:
                completed_indices.add(r['task_index'])
            logger.info(f"Loaded {len(results)} existing results")

    for idx, task in enumerate(tasks):
        if idx in completed_indices:
            logger.info(f"Skipping task {idx + 1}/{len(tasks)} (already processed)")
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"Task {idx + 1}/{len(tasks)}")
        logger.info(f"{'='*80}")

        result = await evaluate_one_task(
            question=task['ques'],
            root_url=task['web'],
            answer=task.get('answer', ''),
            task_idx=idx,
            task_id=task['id'],
            log_dir=log_dir,
            model_name=model_name,
            preprocessed_data=preprocessed_data,
            provider=provider,
            max_turns_per_navigation=max_turns_per_navigation,
            max_total_navigations=max_total_navigations,
        )

        result['task_index'] = idx
        result['item_info'] = task
        results.append(result)
        completed_indices.add(idx)

        sorted_results = sorted(results, key=lambda x: x['task_index'])
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_results, f, indent=4, ensure_ascii=False)

    successful = sum(1 for r in results if r['success'])
    total = len(results)
    if total > 0:
        logger.info(f"No-memory ablation with {model_name}: {successful}/{total} ({successful/total*100:.1f}%)")

    root_logger.removeHandler(file_handler)
    root_logger.removeHandler(console_handler)
    file_handler.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="evaluation_results")
    parser.add_argument("--model", type=str, default="qwen3-32b")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-navs", type=int, default=10)

    args = parser.parse_args()

    asyncio.run(main(
        limit=args.limit,
        output_dir=args.output,
        model_name=args.model,
        provider=args.provider,
        max_turns_per_navigation=args.max_turns,
        max_total_navigations=args.max_navs,
    ))
