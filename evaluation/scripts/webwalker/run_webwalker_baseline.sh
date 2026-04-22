#!/bin/bash

python evaluation/scripts/webwalker/run_webwalker_baseline.py --model qwen3-4b --max-rounds 100
python evaluation/scripts/webwalker/run_webwalker_baseline.py --model qwen3-8b --max-rounds 100
python evaluation/scripts/webwalker/run_webwalker_baseline.py --model qwen3-14b --max-rounds 100
python evaluation/scripts/webwalker/run_webwalker_baseline.py --model qwen3-32b --max-rounds 100
python evaluation/scripts/webwalker/run_webwalker_baseline.py --model gpt-4.1 --max-rounds 100
