import asyncio
import json
import logging
import sys
import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../baselines/webwalker')))

from evaluation.baselines.webwalker.usage import webwalker_navigate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def evaluate_one_task_async(
    question: str,
    root_url: str,
    answer: str,
    task_idx: int,
    log_dir: str,
    llm_config: dict,
    max_rounds: int = 10
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
    logger.info(f"Max Rounds: {max_rounds}")

    try:
        loop = asyncio.get_event_loop()
        result_answer = await loop.run_in_executor(
            None,
            webwalker_navigate,
            root_url,
            question,
            llm_config.copy(),
            max_rounds
        )

        logger.info(f"WebWalker Answer: {result_answer}")

        return {
            "question": question,
            "root_url": root_url,
            "expected_answer": answer,
            "answer": result_answer,
            "success": result_answer is not None and result_answer != "No answer found"
        }

    except Exception as e:
        logger.error(f"Error evaluating task: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "question": question,
            "root_url": root_url,
            "expected_answer": answer,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


async def main(
    limit: int = None,
    output_dir: str = "evaluation_results",
    model: str = "qwen3-4b",
    max_rounds: int = 10
):
    base_dir = os.path.join(output_dir, "webwalker", "baseline", "webwalker", model)
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.info(f"Loading WebWalkerQA dataset")
    ds = load_dataset("callanwu/WebWalkerQA", split="main")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    logger.info(f"Evaluating {len(ds)} tasks with WebWalker baseline using model {model}")

    if model.startswith("qwen"):
        if 'DASHSCOPE_API_KEY' not in os.environ:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")
        llm_config = {
            'model': model,
            'api_key': os.getenv('DASHSCOPE_API_KEY'),
            'model_server': "https://dashscope.aliyuncs.com/compatible-mode/v1",
            'generate_cfg': {
                'top_p': 0.8,
                'max_input_tokens': 120000,
                'max_retries': 3,
            },
        }
    elif model.startswith("gpt"):
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variables not set")
        llm_config = {
            'model': model,
            'api_key': os.getenv('OPENAI_API_KEY'),
            'model_server': "https://api.openai.com/v1",
            'generate_cfg': {
                'top_p': 0.8,
                'max_input_tokens': 120000,
                'max_retries': 3,
            },
        }
    else:
        raise ValueError(f"Model {model} not supported. Use qwen* or gpt* models.")

    output_file = os.path.join(base_dir, "results.json")
    results = []
    completed_indices = set()

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            completed_indices = {r['task_index'] for r in results}
            logger.info(f"Loaded {len(results)} existing results")

    for idx, item in enumerate(ds):
        root_url = item['root_url']

        if idx in completed_indices:
            logger.info(f"Skipping task {idx + 1}/{len(ds)} (already processed)")
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"Task {idx + 1}/{len(ds)}")
        logger.info(f"{'='*80}")

        result = await evaluate_one_task_async(
            question=item['question'],
            root_url=item['root_url'],
            answer=item['answer'],
            task_idx=idx,
            log_dir=log_dir,
            llm_config=llm_config,
            max_rounds=max_rounds
        )

        result['task_index'] = idx
        results.append(result)
        completed_indices.add(idx)

        sorted_results = sorted(results, key=lambda x: x['task_index'])

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_results, f, indent=4, ensure_ascii=False)

    successful = sum(1 for r in results if r['success'])
    total = len(results)
    logger.info(f"\nWebWalker baseline with {model}: {successful}/{total} successful ({successful/total*100:.1f}%)")
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="../../../evaluation_results")
    parser.add_argument("--model", type=str, default="qwen3-4b")
    parser.add_argument("--max-rounds", type=int, default=100)

    args = parser.parse_args()

    asyncio.run(main(
        limit=args.limit,
        output_dir=args.output,
        model=args.model,
        max_rounds=args.max_rounds
    ))
