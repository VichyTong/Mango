from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import logging

from agents import Agent, Runner, trace, function_tool
from agents.exceptions import MaxTurnsExceeded
from agents.items import ItemHelpers, TResponseInputItem
from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings
from agents.mcp import MCPServerStdio

from llm_web_scraper.utils.llm import count_tokens

logger = logging.getLogger(__name__)
from llm_web_scraper.prompts.simple_navigation import (
    get_navigation_instruction,
    get_navigation_prompt,
)
from llm_web_scraper.prompts.reflection import (
    get_reflection_success_instruction,
    get_reflection_failure_instruction,
    get_function_list,
    ReflectionOutput,
    ReflectionSuccessOutput,
    ReflectionFailureOutput,
)


@dataclass
class NavigationResult:
    trace_id: str
    final_output: ReflectionOutput
    function_list: List[tuple[str, Any]]
    tokens_used: int


def construct_input(existing_history: List[Dict[str, Any]]) -> TResponseInputItem:
    if not existing_history:
        return None

    content_parts = ["## Navigation History"]

    for try_number, nav_item in enumerate(existing_history, start=1):
        content_parts.append(f"\n### Try {try_number}")

        function_calls = nav_item.get("function_calls", [])
        if function_calls:
            content_parts.append("**Tool calls:**")
            for func_name, func_args in function_calls:
                content_parts.append(f"- {func_name} with args: {func_args}")

        reflection = nav_item.get("reflection")
        if reflection:
            content_parts.append("**Reflection:**")
            content_parts.append("- status: " + reflection.status)
            content_parts.append("- reason: " + reflection.reason)
            if hasattr(reflection, 'extracted_content'):
                content_parts.append("- extracted_content: " + reflection.extracted_content)
            if hasattr(reflection, 'source'):
                content_parts.append("- source: " + reflection.source)
    content = "\n".join(content_parts)

    return {
        "role": "user",
        "content": content
    } 


async def web_navigation(
    user_intent: str,
    url: str,
    max_turns: int,
    model: LitellmModel,
    model_settings: ModelSettings,
    existing_history: Optional[List[Dict[str, Any]]] = [],
    metadata: Optional[dict[str, Any]] = None,
) -> NavigationResult:
    success_reflection_agent = Agent(
        name="Success Reflection Agent",
        instructions=get_reflection_success_instruction(),
        model=model,
        output_type=ReflectionSuccessOutput,
        model_settings=model_settings,
    )

    failure_reflection_agent = Agent(
        name="Failure Reflection Agent",
        instructions=get_reflection_failure_instruction(),
        model=model,
        output_type=ReflectionFailureOutput,
        model_settings=model_settings,
    )

    instruction = get_navigation_instruction()
    new_prompt = [{
        "role": "user",
        "content": get_navigation_prompt(user_intent, url)
    }]

    history_input = construct_input(existing_history)
    if history_input:
        prompt = [history_input] + new_prompt
    else:
        prompt = new_prompt

    server = MCPServerStdio(
        name="Playwright Browser",
        params={
            "command": "npx",
            "args": [
                "-y",
                "@playwright/mcp@latest",
                "--browser",
                "chrome",
                "--isolated",
                "--headless",
                "--no-sandbox",
            ],
        },
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=3,
    )

    agent = Agent(
        name="Web Navigator",
        instructions=instruction,
        model=model,
        mcp_servers=[server],
        handoffs=[success_reflection_agent, failure_reflection_agent],
        model_settings=model_settings,
    )

    await server.connect()

    with trace("Web_Navigation", metadata=metadata) as t:
        trace_id = t.trace_id
        total_tokens = 0

        try:
            result = await Runner.run(agent, prompt, max_turns=max_turns)
            navigation_tokens = count_tokens(result.raw_responses)
            total_tokens += navigation_tokens
            logger.info(f"Web Navigator tokens: {navigation_tokens}")
            if not isinstance(result.final_output, ReflectionOutput):
                input_list = result.to_input_list()
                if "success reflection agent" in result.final_output:
                    try:
                        result = await Runner.run(success_reflection_agent, input_list)
                        reflection_tokens = count_tokens(result.raw_responses)
                        total_tokens += reflection_tokens
                        logger.info(f"Success Reflection tokens: {reflection_tokens}")
                    except Exception as re:
                        raise re
                elif "failure reflection agent" in result.final_output:
                    try:
                        result = await Runner.run(failure_reflection_agent, input_list)
                        reflection_tokens = count_tokens(result.raw_responses)
                        total_tokens += reflection_tokens
                        logger.info(f"Failure Reflection tokens: {reflection_tokens}")
                    except Exception as re:
                        raise re
                else:
                    raise ValueError("Unexpected final output type from Web Navigator")

        except MaxTurnsExceeded as e:
            original_items: list[TResponseInputItem] = ItemHelpers.input_to_new_input_list(e.run_data.input)
            new_items = [item.to_input_item() for item in e.run_data.new_items]
            input_list = original_items + new_items
            
            try:
                if new_items[-1]["output"] == '{"assistant": "Success Reflection Agent"}':
                    # if the last output indicates success reflection
                    result = await Runner.run(success_reflection_agent, input_list)
                    new_items += result.new_items
                    reflection_tokens = count_tokens(result.raw_responses)
                    total_tokens += reflection_tokens
                    logger.info(f"Success Reflection tokens: {reflection_tokens}")
                else:
                    result = await Runner.run(failure_reflection_agent, input_list)
                    new_items += result.new_items
                    reflection_tokens = count_tokens(result.raw_responses)
                    total_tokens += reflection_tokens
                    logger.info(f"Failure Reflection tokens: {reflection_tokens}")
            except Exception as re:
                raise re
        except Exception as e:
            raise e

        logger.info(f"Total tokens used: {total_tokens}")

    await server.cleanup()

    function_list = get_function_list(result.to_input_list())

    return NavigationResult(
        trace_id=trace_id,
        final_output=result.final_output,
        function_list=function_list,
        tokens_used=total_tokens,
    )
