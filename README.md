# Mango: Multi-Agent Web Navigation via Global-View Optimization

This is the official implementation of [Mango: Multi-Agent Web Navigation via Global-View Optimization](https://arxiv.org/abs/2604.18779) (ACL 2026).

Mango is a web navigation framework that constructs a global view of a website's structure before navigation begins. It identifies query-relevant entry-point URLs using lightweight BFS crawling and Google Search, then models URL selection as a multi-armed bandit problem with Thompson Sampling to allocate the navigation budget efficiently. An episodic memory component prevents the agent from repeating unsuccessful actions across navigation attempts.

## Method Overview

1. **Global Structure Analysis** — BFS-crawls the target website and augments with Google Search results. Scores all candidate URLs with BM25 against the user query.
2. **MAB URL Selection** — Models URL selection as a multi-armed bandit. Initializes Beta distribution priors from BM25 relevance scores, then uses Thompson Sampling to adaptively allocate the navigation budget.
3. **Web Navigation Agent** — Navigates from the selected URL using a browser tool, then hands off to a reflection agent.
4. **Reflection Agent** — Evaluates the navigation trajectory. Updates the bandit posterior and stores the trajectory in episodic memory.

## Installation

```bash
pip install -r requirements.txt
playwright install
```

## Setup

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for GPT models and agent tracing |
| `DASHSCOPE_API_KEY` | Required for Qwen3 models |
| `SEARCH_API_KEY` | Google Custom Search API key (for preprocessing) |
| `SEARCH_ENGINE_ID` | Google Custom Search Engine ID (for preprocessing) |

## Project Structure

```
├── llm_web_scraper/
│   ├── navigator/              # Web navigation agent (Crawl4AI-based)
│   ├── prompts/                # LLM prompts (navigation, reflection, final answer)
│   ├── selector/               # Thompson Sampling and greedy URL selectors
│   └── url_preprocessing/      # URL candidate generation (crawl, Google search, random)
├── evaluation/
│   ├── datasets/               # WebVoyager dataset
│   ├── preprocess/             # Preprocessing scripts
│   └── scripts/                # Evaluation scripts (WebVoyager, WebWalkerQA)
└── requirements.txt
```

## Evaluation

Evaluation runs in two steps: preprocessing (URL candidate generation) then navigation.

### Step 1: Preprocess URL Candidates

Generate the candidate URL sets for each task before running navigation:

```bash
# WebVoyager
bash evaluation/preprocess/webvoyager.sh

# WebWalkerQA
bash evaluation/preprocess/webwalker.sh
```

This runs both the Google Search and BFS crawl preprocessing for all backbone models. Results are saved under `evaluation_results/*/preprocess/`.

To run a single benchmark/method/model:

```bash
python evaluation/preprocess/run_preprocessing.py \
    --benchmark webvoyager \    # webvoyager | webwalker
    --method google \           # google | crawl | random
    --model gpt-5-mini
```

### Step 2: Run Navigation

```bash
# WebVoyager
bash evaluation/scripts/webvoyager/run.sh

# WebWalkerQA
bash evaluation/scripts/webwalker/run.sh
```

To run a specific model or method:

```bash
python evaluation/scripts/webvoyager/run.py \
    --model gpt-5-mini \
    --methods ours google random \
    --navigator simple
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gpt-5-mini` | Backbone LLM |
| `--methods` | `ours google random` | URL selection strategies |
| `--navigator` | `simple` | Browser environment (`simple` = Crawl4AI, `mcp` = Playwright) |

### Step 3: Evaluate Results

```bash
# WebVoyager
bash evaluation/scripts/webvoyager/evaluate.sh

# WebWalkerQA
bash evaluation/scripts/webwalker/evaluate.sh
```

### Running the WebWalker Baseline

```bash
bash evaluation/scripts/webwalker/run_webwalker_baseline.sh
```