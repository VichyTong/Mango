python evaluation/preprocess/run_preprocessing.py --benchmark webwalker --method google --model qwen3-4b
python evaluation/preprocess/run_preprocessing.py --benchmark webwalker --method google --model qwen3-8b
python evaluation/preprocess/run_preprocessing.py --benchmark webwalker --method google --model qwen3-14b
python evaluation/preprocess/run_preprocessing.py --benchmark webwalker --method google --model qwen3-32b
python evaluation/preprocess/run_preprocessing.py --benchmark webwalker --method google --model gpt-5-mini

python evaluation/preprocess/run_preprocessing.py --benchmark webwalker --method crawl --model qwen3-4b
python evaluation/preprocess/run_preprocessing.py --benchmark webwalker --method crawl --model qwen3-8b
python evaluation/preprocess/run_preprocessing.py --benchmark webwalker --method crawl --model qwen3-14b
python evaluation/preprocess/run_preprocessing.py --benchmark webwalker --method crawl --model qwen3-32b
python evaluation/preprocess/run_preprocessing.py --benchmark webwalker --method crawl --model gpt-5-mini

python evaluation/preprocess/run_preprocessing.py --benchmark webwalker --method random
