import os
import time
import requests
from typing import Optional, List
import logging

from agents import Agent, Runner, trace
from agents.model_settings import ModelSettings

from llm_web_scraper.utils.llm import get_litellm_model, create_model_settings
from llm_web_scraper.utils.llm import count_tokens


def google_search(
    query: str,
    num_results: int = 10,
    api_key: Optional[str] = None,
    search_engine_id: Optional[str] = None,
    max_retries: int = 3
) -> List[str]:
    logger = logging.getLogger(__name__)

    if api_key is None:
        api_key = os.getenv('SEARCH_API_KEY')
    if search_engine_id is None:
        search_engine_id = os.getenv('SEARCH_ENGINE_ID')

    if not api_key or not search_engine_id:
        error_msg = "Missing Google Search API credentials. Set SEARCH_API_KEY and SEARCH_ENGINE_ID environment variables."
        logger.error(error_msg)
        raise ValueError(error_msg)

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': search_engine_id,
        'q': query,
        'num': num_results
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json()

            if 'error' in search_results:
                error_info = search_results['error']
                error_msg = f"Google API Error: {error_info.get('message', 'Unknown error')}"
                logger.error(f"Attempt {attempt + 1}/{max_retries}: {error_msg}")
                last_error = error_msg

                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                    continue
                else:
                    raise RuntimeError(error_msg)

            urls = []
            if 'items' in search_results:
                for item in search_results['items']:
                    link = item.get('link')
                    if link:
                        urls.append(link)

            if urls:
                return urls
            else:
                return []

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error {e.response.status_code}: {e.response.text}"
            logger.error(f"Attempt {attempt + 1}/{max_retries}: {error_msg}")
            last_error = error_msg

            if attempt < max_retries - 1:
                import time
                time.sleep(1)
                continue

        except requests.exceptions.RequestException as e:
            error_msg = f"Request Error: {str(e)}"
            logger.error(f"Attempt {attempt + 1}/{max_retries}: {error_msg}")
            last_error = error_msg

            if attempt < max_retries - 1:
                import time
                time.sleep(1)
                continue

        except Exception as e:
            error_msg = f"Unexpected Error: {type(e).__name__}: {str(e)}"
            logger.error(f"Attempt {attempt + 1}/{max_retries}: {error_msg}")
            last_error = error_msg

            if attempt < max_retries - 1:
                import time
                time.sleep(1)
                continue

    raise RuntimeError(f"Google Search API request failed after {max_retries} attempts. Last error: {last_error}")


async def get_google_search_results(
    user_intent: str,
    context_url: str,
    num_results: int,
    model_name: str,
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = 64,
) -> tuple[List[str], int, str]:
    logger = logging.getLogger(__name__)

    model = get_litellm_model(model_name=model_name, provider=provider)
    model_settings = create_model_settings(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

    search_agent = Agent(
        name="Query Keywords Generator",
        instructions="You are a search expert. Transform the user's intent into a concise, 5-word search query composed of space-separated keywords. Output only the final query.",
        model=model,
        model_settings=model_settings,
    )

    prompt_content = f"User Intent: {user_intent}"

    llm_start = time.time()
    with trace("Search_Query_Generation") as t:
        result = await Runner.run(search_agent, prompt_content)
    llm_elapsed = time.time() - llm_start

    tokens_used = count_tokens(result.raw_responses)
    logger.info(f"Query Keywords trace id: {t.trace_id}")
    logger.info(f"Query Keywords Generation tokens: {tokens_used}")
    logger.info(f"[TIMING] LLM keyword generation: {llm_elapsed:.3f}s")

    generated_query = result.final_output
    query = f"{generated_query} site:{context_url}"
    google_start = time.time()
    urls = google_search(query=query, num_results=num_results)
    google_elapsed = time.time() - google_start
    logger.info(f"[TIMING] Google API call: {google_elapsed:.3f}s")

    if context_url not in urls:
        urls.insert(0, context_url)

    return urls, tokens_used, generated_query
