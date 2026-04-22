import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_task_configs(config_dir: str = "evaluation/baselines/AgentOccam/config_files/tasks"):
    task_configs = {}
    config_path = Path(config_dir)

    for config_file in config_path.glob("*.json"):
        with open(config_file, 'r') as f:
            config = json.load(f)
            task_id = config['task_id']
            task_configs[task_id] = config

    return task_configs

def build_evaluation_message(pred: str, reference: str, question: str) -> list:
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student has performed the action to get the answer. They are allowed to use different phrasing or wording to answer the question. The goal is to evaluate whether the key points in the reference answer are included in the student's answer. We allow answers with additional information that doesn't contradict the reference answer and review them as fully (not partially) correct.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by correct/incorrect/partially correct and explain why."

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    return messages

def parse_evaluation_response(response_text: str) -> float:
    response_text = response_text.lower()
    if "partially correct" in response_text or "incorrect" in response_text:
        return 0.0
    else:
        return 1.0

def get_evaluation_data(pred: str, config: dict) -> dict:
    if "eval" not in config or "reference_answers" not in config["eval"]:
        return None

    intent = config.get("intent", "")
    reference_answers = config["eval"]["reference_answers"]

    if "fuzzy_match" not in reference_answers:
        return None

    value = reference_answers["fuzzy_match"]
    if isinstance(value, list):
        reference = "; ".join(value)
    else:
        reference = value

    return {
        "prediction": pred,
        "reference": reference,
        "question": intent
    }

def generate_batch_jsonl(input_path: str, config_dir: str = "evaluation/baselines/AgentOccam/config_files/tasks"):
    task_configs = load_task_configs(config_dir)
    print(f"Loaded {len(task_configs)} task configurations")

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    output_path = input_path.replace(".json", "_batch_requests.jsonl")

    batch_requests = []
    skipped = 0

    for data in input_data:
        task_id = data.get("item_info", {}).get("id") or data.get("id")
        if not task_id:
            skipped += 1
            continue

        if task_id not in task_configs:
            skipped += 1
            continue

        final_output = data.get("final_output")
        if final_output is None:
            skipped += 1
            continue

        config = task_configs[task_id]
        eval_data = get_evaluation_data(str(final_output), config)

        if eval_data is None:
            skipped += 1
            continue

        messages = build_evaluation_message(
            pred=eval_data["prediction"],
            reference=eval_data["reference"],
            question=eval_data["question"]
        )

        batch_request = {
            "custom_id": task_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4-1106-preview",
                "messages": messages,
                "temperature": 0,
                "max_tokens": 768,
                "top_p": 1.0
            }
        }

        batch_requests.append(batch_request)

    with open(output_path, "w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    print(f"Generated {len(batch_requests)} batch requests")
    print(f"Skipped {skipped} items")
    print(f"Batch requests saved to: {output_path}")

    return output_path

def process_batch_results(batch_results_path: str, input_path: str, output_path: str, config_dir: str = "evaluation/baselines/AgentOccam/config_files/tasks"):
    task_configs = load_task_configs(config_dir)

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    input_data_dict = {}
    for data in input_data:
        task_id = data.get("item_info", {}).get("id") or data.get("id")
        if task_id:
            input_data_dict[task_id] = data

    batch_results = {}
    with open(batch_results_path, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result.get("custom_id")
            if custom_id and "response" in result:
                response_body = result["response"]["body"]
                if "choices" in response_body and len(response_body["choices"]) > 0:
                    content = response_body["choices"][0]["message"]["content"]
                    score = parse_evaluation_response(content)
                    batch_results[custom_id] = {
                        "score": score,
                        "reasoning": content
                    }

    evaluated_items = []
    for task_id, data in input_data_dict.items():
        if task_id in batch_results:
            data["evaluation_score"] = batch_results[task_id]["score"]
            data["evaluation_reasoning"] = batch_results[task_id]["reasoning"]
        evaluated_items.append(data)

    evaluated_items = sorted(
        evaluated_items,
        key=lambda x: (x.get("task_index", float('inf')), x.get("task_id", "") or x.get("id", ""))
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluated_items, f, ensure_ascii=False, indent=2)

    llm_evaluated_items = [item for item in evaluated_items if "evaluation_score" in item]

    print(f"\nTotal items: {len(evaluated_items)}")
    print(f"LLM evaluated items: {len(llm_evaluated_items)}")

    if llm_evaluated_items:
        overall_accuracy = sum(item["evaluation_score"] for item in llm_evaluated_items) / len(llm_evaluated_items)
    else:
        overall_accuracy = 0.0

    result = {
        "overall": overall_accuracy,
        "total_evaluated": len(llm_evaluated_items),
        "total_questions": len(evaluated_items)
    }

    if output_path.endswith(".jsonl"):
        report_path = output_path.replace(".jsonl", "_report.json")
    elif output_path.endswith(".json"):
        report_path = output_path.replace(".json", "_report.json")
    else:
        report_path = output_path + "_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_path}")
    print(f"Report saved to: {report_path}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch evaluation for WebVoyager using OpenAI Batch API")
    parser.add_argument("--mode", type=str, choices=["generate", "process"], required=True,
                        help="Mode: 'generate' to create batch requests, 'process' to process batch results")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Input prediction result path (JSON file)")
    parser.add_argument("--output_path", type=str,
                        help="Evaluation output path (only for 'process' mode)")
    parser.add_argument("--batch_results", type=str,
                        help="Path to batch results JSONL file (only for 'process' mode)")
    parser.add_argument("--config_dir", type=str,
                        default="evaluation/baselines/AgentOccam/config_files/tasks",
                        help="Task config directory")
    args = parser.parse_args()

    if args.mode == "generate":
        generate_batch_jsonl(args.input_path, args.config_dir)
    elif args.mode == "process":
        if not args.batch_results or not args.output_path:
            parser.error("--batch_results and --output_path are required for 'process' mode")
        process_batch_results(args.batch_results, args.input_path, args.output_path, args.config_dir)
