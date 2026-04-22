from typing import Any, Dict, List, Optional
import logging

from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings
from agents import Agent, Runner

from llm_web_scraper.navigator.simple_navigation import NavigationResult
from llm_web_scraper.prompts.reflection import (
    ReflectionOutput,
    ReflectionSuccessOutput,
    ReflectionFailureOutput,
)
from llm_web_scraper.utils.bm25 import BM25ContentFilter
from llm_web_scraper.utils.llm import count_tokens
from llm_web_scraper.prompts.final_answer import get_final_answer_instruction, get_final_answer_prompt
from llm_web_scraper.selector.thompson_sampling import URLNavigationState, NavigationOutcome

logger = logging.getLogger(__name__)


def _rank_urls(urls: List[str], user_intent: str, scoring_method: str = "bm25") -> List[str]:
    if len(urls) == 1:
        return urls

    if scoring_method == "embedding":
        from llm_web_scraper.utils.embedding_scorer import calculate_embedding_scores_for_urls
        scores = calculate_embedding_scores_for_urls(urls=urls, query=user_intent)
        logger.info("Using embedding cosine similarity for URL ranking")
    else:
        bm25_filter = BM25ContentFilter(use_stemming=True, language="english")
        scores = bm25_filter.calculate_bm25_scores_for_urls(urls=urls, query=user_intent)
        logger.info("Using BM25 for URL ranking")

    url_scores = list(zip(urls, scores))
    url_scores.sort(key=lambda x: x[1], reverse=True)

    for url, score in url_scores:
        logger.info(f"URL: {url}, score: {score:.4f}")

    return [url for url, _ in url_scores]


def _determine_outcome(reflection_output: ReflectionOutput) -> NavigationOutcome:
    if isinstance(reflection_output, ReflectionSuccessOutput):
        if reflection_output.status == "inadequate":
            return NavigationOutcome.SUCCESS_AND_CONTINUE
        elif reflection_output.status == "adequate":
            return NavigationOutcome.SUCCESS_AND_STOP
        else:
            raise ValueError(f"Invalid status: {reflection_output.status}")
    elif isinstance(reflection_output, ReflectionFailureOutput):
        if reflection_output.status == "feasible":
            return NavigationOutcome.FAILURE_BUT_CONTINUE
        elif reflection_output.status == "infeasible":
            return NavigationOutcome.FAILURE_AND_STOP
        else:
            raise ValueError(f"Invalid status: {reflection_output.status}")
    else:
        raise TypeError("reflection_output must be ReflectionSuccessOutput or ReflectionFailureOutput")


async def greedy_navigation(
    urls: List[str],
    user_intent: str,
    model: LitellmModel,
    model_settings: ModelSettings,
    max_turns_per_navigation: int = 10,
    max_total_navigations: int = 10,
    navigator_type: str = "simple",
    metadata: Optional[Dict[str, Any]] = None,
    scoring_method: str = "bm25",
) -> Dict[str, Any]:
    from llm_web_scraper.navigator import get_navigator

    navigator_func = await get_navigator(navigator_type)

    logger.info("=" * 80)
    logger.info("Starting Greedy Navigation (BM25-ranked, no Thompson Sampling)")
    logger.info("=" * 80)
    logger.info(f"Navigator type: {navigator_type}")
    logger.info(f"Max turns per navigation: {max_turns_per_navigation}")
    logger.info(f"Max total navigations: {max_total_navigations}")

    ranked_urls = _rank_urls(urls, user_intent, scoring_method=scoring_method)

    navigation_states: Dict[str, URLNavigationState] = {
        url: URLNavigationState(url=url, history=[], trace_ids=[])
        for url in urls
    }
    extracted_contents: List[Dict[str, str]] = []

    total_navigations = 0
    total_tokens = 0
    final_result = None
    url_pointer = 0

    while total_navigations < max_total_navigations:
        logger.info("")
        logger.info(f"{'=' * 80}")
        logger.info(f"Navigation iteration {total_navigations + 1}/{max_total_navigations}")
        logger.info(f"{'=' * 80}")

        if url_pointer >= len(ranked_urls):
            logger.info("All URLs exhausted - stopping navigation")
            break

        selected_url = ranked_urls[url_pointer]
        url_pointer += 1

        nav_state = navigation_states[selected_url]
        logger.info(f"Navigating to {selected_url} (rank {url_pointer}/{len(ranked_urls)})")

        try:
            result: NavigationResult = await navigator_func(
                user_intent=user_intent,
                url=selected_url,
                max_turns=max_turns_per_navigation,
                model=model,
                model_settings=model_settings,
                existing_history=nav_state.history,
                metadata=metadata,
            )

            navigation_history = {
                "function_calls": result.function_list,
                "reflection": result.final_output,
            }
            nav_state.history.append(navigation_history)
            nav_state.trace_ids.append(result.trace_id)

            total_navigations += 1
            total_tokens += result.tokens_used

            logger.info(f"Trace ID: {result.trace_id}")
            logger.info(f"Navigation tokens used: {result.tokens_used}")

            outcome = _determine_outcome(result.final_output)
            logger.info(f"Navigation completed with outcome: {outcome.value}")

            if outcome == NavigationOutcome.SUCCESS_AND_CONTINUE:
                extracted_contents.append({
                    "content": result.final_output.extracted_content,
                    "source": result.final_output.source,
                })
            elif outcome == NavigationOutcome.SUCCESS_AND_STOP:
                logger.info("SUCCESS_AND_STOP outcome reached - stopping navigation")
                extracted_contents.append({
                    "content": result.final_output.extracted_content,
                    "source": result.final_output.source,
                })
                break

        except Exception as e:
            logger.error(f"Error during navigation to {selected_url}: {e}")
            raise e

    final_answer_agent = Agent(
        name="Final Answer Agent",
        instructions=get_final_answer_instruction(),
        model=model,
        model_settings=model_settings,
    )
    if len(extracted_contents) > 0:
        final_answer_prompt = get_final_answer_prompt(extracted_contents, user_intent)
        try:
            final_result_run = await Runner.run(final_answer_agent, [final_answer_prompt], max_turns=1)
            final_result_tokens = count_tokens(final_result_run.raw_responses)
            total_tokens += final_result_tokens
            logger.info(f"Final Answer tokens used: {final_result_tokens}")
            final_result = final_result_run.final_output
            final_status = "success"
            logger.info("Final Answer obtained successfully")
        except Exception as e:
            logger.error(f"Error during final answer generation: {e}")
            raise e
    else:
        final_status = "failure"
        final_result = None
        logger.info("No extracted contents - navigation failed")

    logger.info("")
    logger.info("=" * 80)
    logger.info("Greedy Navigation Complete")
    logger.info("=" * 80)
    logger.info(f"Total navigations performed: {total_navigations}")
    logger.info(f"Total navigation tokens used: {total_tokens}")
    logger.info(f"Final result: {'Found' if final_result else 'Not found'}")

    return {
        "ranked_urls": ranked_urls,
        "navigation_states": navigation_states,
        "total_navigations": total_navigations,
        "total_tokens": total_tokens,
        "final_status": final_status,
        "final_result": final_result,
    }
