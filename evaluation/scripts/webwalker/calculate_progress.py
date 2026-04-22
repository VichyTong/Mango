#!/usr/bin/env python3

import os
import json

TOTAL_TASKS = 680

METHODS = [
    "google",
    "ours",
    "random"
]

MODELS = [
    "qwen3-4b",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "gpt-5-mini"
]

BASELINES = [
    "baseline/webwalker",
    "baseline/mcts",
    "baseline/agentoccam"
]

def count_completed_tasks(result_dir):
    result_json = os.path.join(result_dir, "thompson_sampling", "results.json")
    if os.path.exists(result_json):
        with open(result_json) as f:
            data = json.load(f)
            return len(data)
    return 0

def calculate_progress():
    total_runs = len(METHODS) * len(MODELS) + len(BASELINES) * len(MODELS)
    total_tasks_needed = total_runs * TOTAL_TASKS

    print(f"WebWalker Benchmark Progress")
    print("=" * 80)
    print(f"Total runs needed: {total_runs}")
    print(f"Total tasks needed: {total_tasks_needed:,}")
    print("=" * 80)
    print()

    completed_total = 0

    print("Methods × Models:")
    print("-" * 80)
    for method in METHODS:
        for model in MODELS:
            result_dir = f"evaluation_results/webwalker/{model}/{method}"
            completed = count_completed_tasks(result_dir)
            completed_total += completed
            progress = (completed / TOTAL_TASKS) * 100
            status = "✓" if completed == TOTAL_TASKS else "○"
            print(f"{status} {method:12} × {model:15} : {completed:4}/{TOTAL_TASKS} ({progress:6.2f}%)")

    print()
    print("Baselines:")
    print("-" * 80)
    for baseline in BASELINES:
        for model in MODELS:
            result_dir = f"evaluation_results/webwalker/{baseline}/{model}"
            logs_dir = os.path.join(result_dir, "logs")
            if os.path.exists(logs_dir):
                completed = len([f for f in os.listdir(logs_dir) if os.path.isfile(os.path.join(logs_dir, f)) and f.endswith('.log')])
            else:
                completed = 0
            completed_total += completed
            progress = (completed / TOTAL_TASKS) * 100
            status = "✓" if completed == TOTAL_TASKS else "○"
            print(f"{status} {baseline:30} × {model:15} : {completed:4}/{TOTAL_TASKS} ({progress:6.2f}%)")

    print()
    print("=" * 80)
    overall_progress = (completed_total / total_tasks_needed) * 100
    print(f"Overall Progress: {completed_total:,}/{total_tasks_needed:,} ({overall_progress:.2f}%)")
    print(f"Remaining tasks: {total_tasks_needed - completed_total:,}")
    print("=" * 80)

if __name__ == "__main__":
    calculate_progress()
