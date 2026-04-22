import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from llm_web_scraper.selector.thompson_sampling import thompson_sampling_navigation
from llm_web_scraper.utils.llm import get_litellm_model, create_model_settings

logger = logging.getLogger(__name__)


def normalize_model_name_for_path(model_name: str) -> str:
    if model_name.startswith("tensorblock/"):
        return model_name.replace("tensorblock/", "")
    return model_name


def load_preprocessed_urls(method: str, model_name: str, preprocess_dir: str = "evaluation_results/webvoyager/preprocess"):
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

def get_preprocessed_urls_for_task(task_id: str, method: str, preprocessed_data, root_url: str = None) -> tuple[list, int]:
    if method == "ours":
        google_data = preprocessed_data["google"]
        crawl_data = preprocessed_data["crawl"]

        google_urls = []
        google_tokens = 0
        for item in google_data:
            if item['task_id'] == task_id:
                google_urls = item['urls']
                google_tokens = item.get('tokens_used', 0)
                break

        crawl_urls = []
        crawl_tokens = 0
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
        logger.info(f"Task {task_id}: Using crawl ∪ google ({len(crawl_urls)} crawl + {len(google_urls)} google = {len(combined_urls)} unique)")
        return combined_urls, combined_tokens
    else:
        for item in preprocessed_data:
            if item['task_id'] == task_id:
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
    task_id: str,
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

    try:
        urls, url_preprocessing_tokens = get_preprocessed_urls_for_task(task_id, method, preprocessed_data, root_url)

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
                "benchmark": "WebVoyager"
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


async def main(
    methods: list = None,
    limit: int = None,
    output_dir: str = "evaluation_results",
    model_name: str = "gpt-4o-mini",
    provider: str = None,
    navigator: str = "simple"
):
    if methods is None:
        methods = ["ours", "google", "random"]

    normalized_model_name = normalize_model_name_for_path(model_name)
    base_dir = os.path.join(output_dir, "webvoyager", normalized_model_name)
    os.makedirs(base_dir, exist_ok=True)

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

    logger.info(f"Loading WebVoyager dataset")

    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../datasets/webvoyager/WebVoyager_data.jsonl"
    )

    tasks = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    if limit:
        tasks = tasks[:limit]

    logger.info(f"Evaluating {len(tasks)} tasks with methods: {methods}")

    for method in methods:
        method_dir = os.path.join(base_dir, method, "thompson_sampling")
        log_dir = os.path.join(method_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        preprocessed_data = load_preprocessed_urls(method, model_name)
        if method == "ours":
            logger.info(f"Loaded preprocessed URL lists for {method} (google: {len(preprocessed_data['google'])}, crawl: {len(preprocessed_data['crawl'])})")
        else:
            logger.info(f"Loaded {len(preprocessed_data)} preprocessed URL lists for {method}")

        output_file = os.path.join(method_dir, "results.json")
        method_results = []
        completed_indices = set()

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)

                for r in existing_results:
                    task_idx = r['task_index']
                    method_results.append(r)
                    completed_indices.add(task_idx)

                logger.info(f"Loaded {len(existing_results)} existing results for {method}")
                logger.info(f"Completed task indices: {sorted(completed_indices)}")

        for idx, task in enumerate(tasks):
            task_id = task['id']
            root_url = task['web']

            if idx in completed_indices:
                logger.info(f"Skipping task {idx + 1}/{len(tasks)} (already processed)")
                continue

            logger.info(f"\n{'='*80}")
            logger.info(f"Task {idx + 1}/{len(tasks)} - Method: {method}")
            logger.info(f"{'='*80}")

            result = await evaluate_one_task(
                question=task['ques'],
                root_url=root_url,
                answer=task.get('answer', ''),
                method=method,
                task_idx=idx,
                task_id=task_id,
                log_dir=log_dir,
                model_name=model_name,
                preprocessed_data=preprocessed_data,
                provider=provider,
                navigator=navigator
            )

            result['task_index'] = idx
            result['item_info'] = task
            method_results.append(result)
            completed_indices.add(idx)

            sorted_results = sorted(method_results, key=lambda x: x['task_index'])

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sorted_results, f, indent=4, ensure_ascii=False)

        successful = sum(1 for r in method_results if r['success'])
        total = len(method_results)
        logger.info(f"{method} with {model_name}: {successful}/{total} successful ({successful/total*100:.1f}%)")

    logger.info(f"\nEvaluation complete. Results saved to {base_dir}")

    root_logger.removeHandler(file_handler)
    root_logger.removeHandler(console_handler)
    file_handler.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=["ours", "google", "random"],
                        default=["ours", "google", "random"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="evaluation_results")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument(
        "--navigator",
        type=str,
        choices=["simple", "mcp"],
        default="simple",
        help="Navigator type to use: 'simple' (Crawl4AI) or 'mcp' (Playwright MCP)"
    )

    args = parser.parse_args()

    asyncio.run(main(
        methods=args.methods,
        limit=args.limit,
        output_dir=args.output,
        model_name=args.model,
        provider=args.provider,
        navigator=args.navigator
    ))
