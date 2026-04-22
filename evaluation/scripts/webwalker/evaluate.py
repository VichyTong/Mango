import os
import json
import time
import concurrent.futures
from tqdm import tqdm
from datasets import load_dataset
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0,
    api_key=os.getenv('OPENAI_API_KEY'),
)

# Dictionary to store questions, answers, and additional information
info_adic = {}

# Load the dataset
ds = load_dataset("callanwu/WebWalkerQA", split="main")
for question, answer, info in zip(ds["question"], ds["answer"], ds["info"]):
    info_adic[question] = [answer, info]

def eval_result(input_path, output_path):
    """
    Evaluates prediction results against reference answers and generates a report.

    Parameters:
        input_path (str): Path to the input predictions file (JSON array format).
        output_path (str): Path to save the evaluation results and report.
    """
    evaluator = load_evaluator("cot_qa", llm=llm)
    data_list = []
    visited = []

    # Load existing results if output file exists
    existing_results = []
    existing_results_dict = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
                visited = [item["question"] for item in existing_results]
                existing_results_dict = {item["question"]: item for item in existing_results}
        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = []

    # Load and filter data (expecting JSON array format)
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    for data in input_data:
        should_evaluate = False

        if data["question"] not in visited:
            should_evaluate = True
        else:
            existing_item = existing_results_dict[data["question"]]
            current_prediction = data.get("answer") or data.get("final_output")
            existing_prediction = existing_item.get("answer") or existing_item.get("final_output")

            if current_prediction != existing_prediction:
                should_evaluate = True
                existing_results = [item for item in existing_results if item["question"] != data["question"]]
                visited.remove(data["question"])

        if should_evaluate:
            expected_answer = info_adic.get(data["question"], [None, None])[0]
            if expected_answer is not None:
                data["expected_answer"] = expected_answer
                data_list.append(data)

    def call(data):
        """Handles evaluation retries with exponential backoff."""
        max_retries = 10
        for attempt in range(max_retries):
            try:
                try:
                    if "answer" in data:
                        prediction = data["answer"]
                    elif "final_output" in data:
                        if data["final_output"] is None:
                            raise ValueError("final_output is None")
                        prediction = data["final_output"]
                    else:
                        raise ValueError("No valid prediction field found")
                except Exception as e:
                    return {
                        "reasoning": f"No final output found: {e}",
                        "value": "INCORRECT",
                        "score": 0
                    }
                return evaluator.evaluate_strings(
                    prediction=prediction,
                    input=data["question"],
                    reference=data["expected_answer"]
                )
            except Exception as e:
                print(f"Error during evaluation: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (2 ** attempt))  # Exponential backoff
                else:
                    raise e  # Raise the exception if the last retry fails

    s = 0
    cnt = 0
    evaluated_items = existing_results.copy()  # Start with existing results

    with tqdm(total=len(data_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_data = {executor.submit(call, data): data for data in data_list}
            for future in concurrent.futures.as_completed(future_to_data):
                try:
                    outputs = future.result(timeout=4)
                    data = future_to_data[future]

                    # Add evaluation score to the data structure
                    data["evaluation_score"] = outputs["score"]
                    data["evaluation_reasoning"] = outputs["reasoning"]

                    # Add item_info from dataset if not present
                    if "item_info" not in data and data["question"] in info_adic:
                        data["item_info"] = info_adic[data["question"]][1]

                    cnt += outputs["score"]
                    s += 1

                    # Add evaluated item to results
                    evaluated_items.append(data)

                    # Sort by task_index if available, otherwise by question
                    evaluated_items = sorted(
                        evaluated_items,
                        key=lambda x: (x.get("task_index", float('inf')), x.get("question", ""))
                    )

                    # Save updated results as JSON array
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(evaluated_items, f, ensure_ascii=False, indent=2)

                    pbar.update(1)
                    print("Current accuracy:", cnt / s)

                except Exception as e:
                    print(f"Error processing data: {e}")

    # Prepare statistics for the report
    single_source_easy, single_source_medium, single_source_hard = [], [], []
    multi_source_easy, multi_source_medium, multi_source_hard = [], [], []
    overall = []

    # Filter to only items that have been evaluated by LLM (have evaluation_score)
    llm_evaluated_items = [item for item in evaluated_items if "evaluation_score" in item]

    print(f"\nTotal items: {len(evaluated_items)}")
    print(f"LLM evaluated items: {len(llm_evaluated_items)}")

    for temp in llm_evaluated_items:
        # Use LLM evaluation_score
        score = temp.get("evaluation_score")
        if score is not None:
            # Get item_info from the data structure
            info = temp.get("item_info", {})
            q_type = info.get("type")
            difficulty = info.get("difficulty_level")

            if q_type == "single_source":
                if difficulty == "easy":
                    single_source_easy.append(score)
                elif difficulty == "medium":
                    single_source_medium.append(score)
                elif difficulty == "hard":
                    single_source_hard.append(score)

            elif q_type == "multi_source":
                if difficulty == "easy":
                    multi_source_easy.append(score)
                elif difficulty == "medium":
                    multi_source_medium.append(score)
                elif difficulty == "hard":
                    multi_source_hard.append(score)

            overall.append(score)

    # Safely compute averages to avoid division by zero
    def safe_average(scores):
        return sum(scores) / len(scores) if scores else None

    result = {
        "single_source_easy": safe_average(single_source_easy),
        "single_source_medium": safe_average(single_source_medium),
        "single_source_hard": safe_average(single_source_hard),
        "multi_source_easy": safe_average(multi_source_easy),
        "multi_source_medium": safe_average(multi_source_medium),
        "multi_source_hard": safe_average(multi_source_hard),
        "overall": safe_average(overall),
        "total_evaluated": len(llm_evaluated_items),
        "total_questions": len(evaluated_items)
    }

    # Save the report (handle both .jsonl and .json extensions)
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
    print(f"Overall accuracy: {safe_average(overall):.4f}" if overall else "No items evaluated")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Input prediction result path")
    parser.add_argument("--output_path", type=str, help="Evaluation output path")
    args = parser.parse_args()

    eval_result(args.input_path, args.output_path)