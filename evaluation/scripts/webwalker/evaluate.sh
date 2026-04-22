models=("qwen3-4b" "qwen3-8b" "qwen3-14b" "qwen3-32b" "gpt-4.1")
methods=("ours" "google" "random")

for model in "${models[@]}"; do
  for method in "${methods[@]}"; do
    input_path="evaluation_results/webwalker/${model}/${method}/thompson_sampling/results.json"
    output_path="evaluation_results/webwalker/${model}/${method}/thompson_sampling/results_evaluated.json"

    if [ -f "$input_path" ]; then
      echo "Evaluating ${model} - ${method}..."
      python evaluation/scripts/webwalker/evaluate.py --input_path "$input_path" --output_path "$output_path"
    else
      echo "Skipping ${model} - ${method} (file not found: $input_path)"
    fi
  done
done

for model in "${models[@]}"; do
  input_path="evaluation_results/webwalker/baseline/webwalker/${model}/results.json"
  output_path="evaluation_results/webwalker/baseline/webwalker/${model}/results_evaluated.json"

  if [ -f "$input_path" ]; then
    echo "Evaluating baseline webwalker - ${model}..."
    python evaluation/scripts/webwalker/evaluate.py --input_path "$input_path" --output_path "$output_path"
  else
    echo "Skipping baseline webwalker - ${model} (file not found: $input_path)"
  fi
done
