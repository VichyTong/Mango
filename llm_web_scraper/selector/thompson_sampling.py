from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Dict, Optional
from agents import Agent, Runner, trace
import numpy as np
import logging

from agents.items import TResponseInputItem
from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings

from llm_web_scraper.navigator.simple_navigation import NavigationResult
from llm_web_scraper.prompts.reflection import (
    ReflectionOutput,
    ReflectionSuccessOutput,
    ReflectionFailureOutput,
)
from llm_web_scraper.utils.bm25 import BM25ContentFilter
from llm_web_scraper.utils.llm import count_tokens
from llm_web_scraper.prompts.final_answer import get_final_answer_instruction, get_final_answer_prompt

logger = logging.getLogger(__name__)


class NavigationOutcome(Enum):
    SUCCESS_AND_CONTINUE = "inadequate"
    SUCCESS_AND_STOP = "adequate"
    FAILURE_BUT_CONTINUE = "feasible"
    FAILURE_AND_STOP = "infeasible"


@dataclass
class ArmState:
    url: str
    alpha: float
    beta: float
    original_similarity: float
    exhausted: bool = False


@dataclass
class URLNavigationState:
    """Track navigation state for each URL to enable continuation"""
    url: str
    history: List[TResponseInputItem]
    trace_ids: List[str]


class SemanticThompsonSelector:
    def __init__(self, use_stemming: bool = True, language: str = "english", kappa: float = 3.0):
        self.arms: Dict[str, ArmState] = {}
        self.base_prior = 1.0
        self.scaling_factor = kappa
        self.use_stemming = use_stemming
        self.language = language

    def initialize_arms(self, urls: List[str], user_intent: str, scoring_method: str = "bm25") -> None:
        logger.info(f"Initializing Thompson Sampling arms for {len(urls)} URLs")
        logger.info(f"User intent: {user_intent}")
        logger.info(f"Scoring method: {scoring_method}")

        if len(urls) == 1:
            logger.info("Only 1 URL found, skipping scoring")
            similarity = 1.0
            alpha = self.base_prior + similarity * self.scaling_factor
            beta = self.base_prior + (1.0 - similarity) * self.scaling_factor

            logger.info(f"URL: {urls[0]}")
            logger.info(f"  Normalized similarity: {similarity:.4f}")
            logger.info(f"  Initial alpha: {alpha:.4f}, beta: {beta:.4f}")

            self.arms[urls[0]] = ArmState(
                url=urls[0],
                alpha=alpha,
                beta=beta,
                original_similarity=similarity,
                exhausted=False
            )
            return

        if scoring_method == "embedding":
            logger.info("Using embedding cosine similarity for URL scoring")
            from llm_web_scraper.utils.embedding_scorer import calculate_embedding_scores_for_urls
            scores = calculate_embedding_scores_for_urls(urls=urls, query=user_intent)
        else:
            logger.info("Using BM25 for URL scoring")
            bm25_filter = BM25ContentFilter(
                use_stemming=self.use_stemming,
                language=self.language
            )
            scores = bm25_filter.calculate_bm25_scores_for_urls(
                urls=urls,
                query=user_intent
            )

        min_score = min(scores)
        max_score = max(scores)

        for url, score in zip(urls, scores):
            similarity = (score - min_score) / (max_score - min_score + 1e-10)

            alpha = self.base_prior + similarity * self.scaling_factor
            beta = self.base_prior + (1.0 - similarity) * self.scaling_factor

            logger.info(f"URL: {url}")
            logger.info(f"  BM25 score: {score:.4f}")
            logger.info(f"  Normalized similarity: {similarity:.4f}")
            logger.info(f"  Initial alpha: {alpha:.4f}, beta: {beta:.4f}")

            self.arms[url] = ArmState(
                url=url,
                alpha=alpha,
                beta=beta,
                original_similarity=similarity,
                exhausted=False
            )

    def select_next_url(self) -> Optional[str]:
        active_arms = {url: arm for url, arm in self.arms.items() if not arm.exhausted}

        if not active_arms:
            logger.info("No active arms remaining - all URLs exhausted")
            return None

        logger.info(f"Selecting next URL from {len(active_arms)} active arms")
        best_url = None
        best_sample = -np.inf

        for url, arm in active_arms.items():
            sample = np.random.beta(arm.alpha, arm.beta)
            logger.debug(f"  {url}: sample={sample:.4f} (alpha={arm.alpha:.4f}, beta={arm.beta:.4f})")

            if sample > best_sample:
                best_sample = sample
                best_url = url

        logger.info(f"Selected URL: {best_url} with sample={best_sample:.4f}")
        return best_url

    def update_arm(self, url: str, outcome: NavigationOutcome) -> None:
        if url not in self.arms:
            raise ValueError(f"URL '{url}' not found in arms")

        arm = self.arms[url]
        old_alpha = arm.alpha
        old_beta = arm.beta

        logger.info(f"Updating arm for {url}")
        logger.info(f"  Outcome: {outcome.value}")
        logger.info(f"  Before: alpha={old_alpha:.4f}, beta={old_beta:.4f}")

        if outcome == NavigationOutcome.SUCCESS_AND_CONTINUE:
            arm.alpha += 1.0
            arm.exhausted = False
        elif outcome == NavigationOutcome.SUCCESS_AND_STOP:
            arm.alpha += 1.0
            arm.exhausted = True
        elif outcome == NavigationOutcome.FAILURE_BUT_CONTINUE:
            arm.alpha += 1.0
            arm.exhausted = False
        elif outcome == NavigationOutcome.FAILURE_AND_STOP:
            arm.beta += 1.0
            arm.exhausted = True
        else:
            raise ValueError(f"Invalid outcome: {outcome}")

        logger.info(f"  After: alpha={arm.alpha:.4f}, beta={arm.beta:.4f}, exhausted={arm.exhausted}")


async def thompson_sampling_navigation(
    urls: List[str],
    user_intent: str,
    model: LitellmModel,
    model_settings: ModelSettings,
    max_turns_per_navigation: int = 10,
    max_total_navigations: int = 10,
    navigator_type: str = "simple",
    metadata: Optional[dict[str, Any]] = None,
    use_memory: bool = True,
    scoring_method: str = "bm25",
    kappa: float = 3.0,
) -> Dict[str, any]:
    from llm_web_scraper.navigator import get_navigator

    navigator_func = await get_navigator(navigator_type)

    logger.info("=" * 80)
    logger.info("Starting Thompson Sampling Navigation")
    logger.info("=" * 80)
    logger.info(f"Navigator type: {navigator_type}")
    logger.info(f"Max turns per navigation: {max_turns_per_navigation}")
    logger.info(f"Max total navigations: {max_total_navigations}")
    logger.info(f"Use memory: {use_memory}")

    selector = SemanticThompsonSelector(kappa=kappa)
    selector.initialize_arms(urls, user_intent, scoring_method=scoring_method)

    navigation_states: Dict[str, URLNavigationState] = {
        url: URLNavigationState(
            url=url,
            history=[],
            trace_ids=[]
        )
        for url in urls
    }
    extracted_contents: list[Dict[str, str]] = []

    total_navigations = 0
    total_tokens = 0
    final_result = None

    while total_navigations < max_total_navigations:
        logger.info("")
        logger.info(f"{'=' * 80}")
        logger.info(f"Navigation iteration {total_navigations + 1}/{max_total_navigations}")
        logger.info(f"{'=' * 80}")

        selected_url = selector.select_next_url()

        if selected_url is None:
            logger.info("No URL selected - stopping navigation")
            break

        nav_state = navigation_states[selected_url]

        logger.info(f"Navigating to {selected_url} (attempt #{len(nav_state.trace_ids) + 1} for this URL)")

        try:
            result: NavigationResult = await navigator_func(
                user_intent=user_intent,
                url=selected_url,
                max_turns=max_turns_per_navigation,
                model=model,
                model_settings=model_settings,
                existing_history=nav_state.history if use_memory else [],
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

            selector.update_arm(selected_url, outcome)

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
        logger.info("No extracted contents available to generate final answer")



    logger.info("")
    logger.info("=" * 80)
    logger.info("Thompson Sampling Navigation Complete")
    logger.info("=" * 80)
    logger.info(f"Total navigations performed: {total_navigations}")
    logger.info(f"Total navigation tokens used: {total_tokens}")
    logger.info(f"Final result: {'Found' if final_result else 'Not found'}")

    return {
        "selector": selector,
        "navigation_states": navigation_states,
        "total_navigations": total_navigations,
        "total_tokens": total_tokens,
        "final_status": final_status,
        "final_result": final_result,
    }


def _determine_outcome(reflection_output: ReflectionOutput) -> NavigationOutcome:
    if isinstance(reflection_output, ReflectionSuccessOutput):
        if reflection_output.status == "inadequate":
            return NavigationOutcome.SUCCESS_AND_CONTINUE
        elif reflection_output.status == "adequate":
            return NavigationOutcome.SUCCESS_AND_STOP
        else:
            raise ValueError(f"Invalid status in ReflectionSuccessOutput: {reflection_output.status}")
    elif isinstance(reflection_output, ReflectionFailureOutput):
        if reflection_output.status == "feasible":
            return NavigationOutcome.FAILURE_BUT_CONTINUE
        elif reflection_output.status == "infeasible":
            return NavigationOutcome.FAILURE_AND_STOP
        else:
            raise ValueError(f"Invalid status in ReflectionFailureOutput: {reflection_output.status}")
    else:
        raise TypeError("reflection_output must be either ReflectionSuccessOutput or ReflectionFailureOutput")