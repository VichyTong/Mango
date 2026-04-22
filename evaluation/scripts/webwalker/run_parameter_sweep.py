import asyncio
import json
import logging
import sys
import os
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from llm_web_scraper.selector.thompson_sampling import thompson_sampling_navigation
from llm_web_scraper.utils.llm import get_litellm_model, create_model_settings

logger = logging.getLogger(__name__)

PARAM_COMBINATIONS = [
    (5, 10),
    (10, 5),
    (10, 10),
    (10, 15),
    (10, 20),
    (15, 10),
    (20, 10),
]


def normalize_model_name_for_path(model_name: str) -> str:
    if model_name.startswith("tensorblock/"):
        return model_name.replace("tensorblock/", "")
    return model_name


def load_preprocessed_urls(method: str, model_name: str, preprocess_dir: str = "evaluation_results/webwalker/preprocess"):
    normalized_model_name = normalize_model_name_for_path(model_name)

    if method == "random":
        url_list_path = os.path.join(preprocess_dir, method, "url_lists.json")
        with open(url_list_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif method == "ours":
        google_path = os.path.join(preprocess_dir, "google", normalized_model_name, "url_lists.json")
        crawl_path = os.path.join(preprocess_dir, "crawl", normalized_model_name, "url_lists.json")

        with open(google_path, 'r', encoding='utf-8') as f:
            google_data = json.load(f)
        with open(crawl_path, 'r', encoding='utf-8') as f:
            crawl_data = json.load(f)

        return {"google": google_data, "crawl": crawl_data}
    else:
        url_list_path = os.path.join(preprocess_dir, method, normalized_model_name, "url_lists.json")
        with open(url_list_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def filter_file_urls(urls: list) -> list:
    file_extensions = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.rar', '.tar', '.gz', '.7z',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv',
        '.txt', '.csv', '.xml', '.json'
    }
    filtered_urls = []
    for url in urls:
        url_lower = url.lower()
        if not any(url_lower.endswith(ext) for ext in file_extensions):
            filtered_urls.append(url)
    return filtered_urls

def get_preprocessed_urls_for_task(task_idx: int, method: str, preprocessed_data, root_url: str = None) -> tuple[list, int]:
    if method == "ours":
        google_data = preprocessed_data["google"]
        crawl_data = preprocessed_data["crawl"]

        google_urls = []
        google_tokens = 0
        for item in google_data:
            if item['task_index'] == task_idx:
                google_urls = item['urls']
                google_tokens = item.get('tokens_used', 0)
                break

        crawl_urls = []
        crawl_tokens = 0
        for item in crawl_data:
            if item['task_index'] == task_idx:
                crawl_urls = item['urls']
                crawl_tokens = item.get('tokens_used', 0)
                break

        if not google_urls and root_url:
            google_urls = [root_url]

        combined_urls = list(set(crawl_urls) | set(google_urls))
        combined_urls = filter_file_urls(combined_urls)
        combined_tokens = crawl_tokens + google_tokens
        logger.info(f"Task {task_idx}: Using crawl ∪ google ({len(crawl_urls)} crawl + {len(google_urls)} google = {len(combined_urls)} unique)")
        return combined_urls, combined_tokens
    else:
        for item in preprocessed_data:
            if item['task_index'] == task_idx:
                urls = filter_file_urls(item['urls'])
                if not urls and root_url:
                    urls = [root_url]
                return urls, item.get('tokens_used', 0)
        return [], 0


async def evaluate_one_task(
    question: str,
    root_url: str,
    answer: str,
    method: str,
    task_idx: int,
    log_dir: str,
    model_name: str,
    preprocessed_data: list,
    provider: str = None,
    navigator: str = "simple",
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
    logger.info(f"Method: {method}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Max turns per navigation: {max_turns_per_navigation}")
    logger.info(f"Max total navigations: {max_total_navigations}")

    try:
        urls, url_preprocessing_tokens = get_preprocessed_urls_for_task(task_idx, method, preprocessed_data, root_url)

        if not urls:
            logger.warning(f"No URLs found for {root_url} using {method}")
            return {
                "question": question,
                "root_url": root_url,
                "expected_answer": answer,
                "method": method,
                "success": False,
                "error": "No URLs found",
                "urls_count": 0,
                "url_preprocessing_tokens": url_preprocessing_tokens,
                "navigation_tokens": 0,
                "total_tokens": url_preprocessing_tokens
            }

        logger.info(f"Found {len(urls)} URLs using {method}")

        model = get_litellm_model(model_name=model_name, provider=provider)
        model_settings = create_model_settings(model_name=model_name)

        result = await thompson_sampling_navigation(
            urls=urls,
            user_intent=question,
            navigator_type=navigator,
            model=model,
            model_settings=model_settings,
            max_turns_per_navigation=max_turns_per_navigation,
            max_total_navigations=max_total_navigations,
            metadata={
                "model_name": model_name,
                "method": method,
                "task_index": str(task_idx),
                "benchmark": "WebWalkerQA"
            }
        )

        final_output = result.get("final_result", None)
        navigation_tokens = result.get("total_tokens", 0)
        total_tokens = url_preprocessing_tokens + navigation_tokens

        logger.info(f"URL preprocessing tokens: {url_preprocessing_tokens}")
        logger.info(f"Navigation tokens: {navigation_tokens}")
        logger.info(f"Total tokens: {total_tokens}")

        return {
            "question": question,
            "root_url": root_url,
            "expected_answer": answer,
            "method": method,
            "success": result["final_status"] == "success",
            "total_navigations": result["total_navigations"],
            "urls_count": len(urls),
            "final_output": final_output,
            "url_preprocessing_tokens": url_preprocessing_tokens,
            "navigation_tokens": navigation_tokens,
            "total_tokens": total_tokens
        }

    except Exception as e:
        error_str = str(e)
        if "Budget limit exceeded" in error_str or "Insufficient balance" in error_str:
            logger.error(f"Budget limit exceeded - shutting down program")
            root_logger.removeHandler(file_handler)
            file_handler.close()
            sys.exit(1)

        logger.error(f"Error evaluating task: {e}")
        import traceback
        result = {
            "question": question,
            "root_url": root_url,
            "expected_answer": answer,
            "method": method,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()

    return result


async def run_parameter_combination(
    max_turns: int,
    max_navs: int,
    ds,
    task_indices: list,
    preprocessed_data,
    model_name: str,
    method: str,
    navigator: str,
    provider: str,
    output_dir: str,
):
    combination_logger = logging.getLogger(f"combination_{max_turns}_{max_navs}")
    combination_logger.setLevel(logging.INFO)

    normalized_model_name = normalize_model_name_for_path(model_name)
    combination_dir = os.path.join(output_dir, "webwalker", "extra", normalized_model_name, f"turns_{max_turns}_navs_{max_navs}", "thompson_sampling")
    log_dir = os.path.join(combination_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    main_log_file = os.path.join(combination_dir, "run.log")
    file_handler = logging.FileHandler(main_log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(f'[turns={max_turns},navs={max_navs}] %(asctime)s - %(levelname)s - %(message)s'))

    combination_logger.addHandler(file_handler)
    combination_logger.addHandler(console_handler)

    combination_logger.info(f"{'='*100}")
    combination_logger.info(f"Starting parameter combination: max_turns_per_navigation={max_turns}, max_total_navigations={max_navs}")
    combination_logger.info(f"{'='*100}")

    results_file = os.path.join(combination_dir, "results.json")
    combination_results = []
    completed_indices = set()

    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            combination_results = json.load(f)
            for r in combination_results:
                completed_indices.add(r['task_index'])
            combination_logger.info(f"Loaded {len(combination_results)} existing results")

    remaining_indices = [idx for idx in task_indices if idx not in completed_indices]
    combination_logger.info(f"Remaining tasks: {len(remaining_indices)}/{len(task_indices)}")

    for i, actual_idx in enumerate(remaining_indices):
        item = ds[task_indices.index(actual_idx)]

        combination_logger.info(f"Task {actual_idx} ({i+1}/{len(remaining_indices)})")

        result = await evaluate_one_task(
            question=item['question'],
            root_url=item['root_url'],
            answer=item['answer'],
            method=method,
            task_idx=actual_idx,
            log_dir=log_dir,
            model_name=model_name,
            preprocessed_data=preprocessed_data,
            provider=provider,
            navigator=navigator,
            max_turns_per_navigation=max_turns,
            max_total_navigations=max_navs,
        )

        result['task_index'] = actual_idx
        result['item_info'] = item.get('info', {})
        combination_results.append(result)

        sorted_results = sorted(combination_results, key=lambda x: x['task_index'])

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_results, f, indent=4, ensure_ascii=False)

    successful = sum(1 for r in combination_results if r['success'])
    total = len(combination_results)
    combination_logger.info(f"Combination (turns={max_turns}, navs={max_navs}): {successful}/{total} successful ({successful/total*100:.1f}%)")

    combination_logger.removeHandler(file_handler)
    combination_logger.removeHandler(console_handler)
    file_handler.close()

    return {
        "max_turns": max_turns,
        "max_navs": max_navs,
        "successful": successful,
        "total": total,
        "success_rate": successful/total*100 if total > 0 else 0
    }


async def main(model_name: str = "gpt-5-mini"):
    method = "ours"
    limit = 100
    navigator = "simple"
    provider = None
    output_dir = "evaluation_results"

    normalized_model_name = normalize_model_name_for_path(model_name)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    logger.info(f"Model: {model_name}")
    logger.info(f"Method: {method}")
    logger.info(f"Navigator: {navigator}")
    logger.info(f"Task limit: {limit}")
    logger.info(f"Parameter combinations: {len(PARAM_COMBINATIONS)}")
    logger.info(f"Loading WebWalkerQA dataset")
    ds = load_dataset("callanwu/WebWalkerQA", split="main")

    task_indices = list(range(len(ds)))
    if limit:
        task_indices = task_indices[:limit]
        ds = ds.select(task_indices)

    logger.info(f"Dataset loaded: {len(ds)} tasks")
    logger.info(f"Loading preprocessed URLs for method: {method}")

    preprocessed_data = load_preprocessed_urls(method, model_name)
    logger.info(f"Loaded preprocessed URL lists for {method} (google: {len(preprocessed_data['google'])}, crawl: {len(preprocessed_data['crawl'])})")

    logger.info(f"\n{'='*100}")
    logger.info(f"Starting parallel execution of {len(PARAM_COMBINATIONS)} parameter combinations")
    logger.info(f"{'='*100}\n")

    tasks = []
    for max_turns, max_navs in PARAM_COMBINATIONS:
        task = run_parameter_combination(
            max_turns=max_turns,
            max_navs=max_navs,
            ds=ds,
            task_indices=task_indices,
            preprocessed_data=preprocessed_data,
            model_name=model_name,
            method=method,
            navigator=navigator,
            provider=provider,
            output_dir=output_dir,
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    logger.info(f"\n{'='*100}")
    logger.info(f"Parameter sweep complete! Results saved to {output_dir}/webwalker/extra/{normalized_model_name}/")
    logger.info(f"{'='*100}\n")
    logger.info("Summary of all combinations:")
    for result in results:
        logger.info(f"  turns={result['max_turns']}, navs={result['max_navs']}: {result['successful']}/{result['total']} ({result['success_rate']:.1f}%)")

    root_logger.removeHandler(console_handler)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run parameter sweep experiment for WebWalker")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model name (e.g., gpt-5-mini, qwen3-32b)")

    args = parser.parse_args()

    asyncio.run(main(model_name=args.model))
