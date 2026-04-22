models=("qwen3-4b" "qwen3-8b" "qwen3-14b" "qwen3-32b" "gpt-4.1")
methods=("ours" "google" "random")

for model in "${models[@]}"; do
  for method in "${methods[@]}"; do
    input_path="evaluation_results/webvoyager/${model}/${method}/thompson_sampling/results.json"
    output_path="evaluation_results/webvoyager/${model}/${method}/thompson_sampling/results_evaluated.json"

    if [ -f "$input_path" ]; then
      echo "Evaluating ${model} - ${method}..."
      python evaluation/scripts/webvoyager/evaluate.py --input_path "$input_path" --output_path "$output_path"
    else
      echo "Skipping ${model} - ${method} (file not found: $input_path)"
    fi
  done
done
