from typing import List, Dict
import json


def get_final_answer_instruction() -> str:
    return """\
Your task is to synthesize information from one or more extracted content pieces and provide a comprehensive and accurate response to the user's intent.

## Task Instructions
1. Review all extracted content pieces and their sources.
2. Produce a clear and well-structured answer based on the extracted content.
"""


def get_final_answer_prompt(extracted_contents: List[Dict[str, str]], user_intent: str) -> Dict[str, str]:
    return {
        "role": "user",
        "content": f"""\
## User Intent
{user_intent}

## Extracted Content
{json.dumps(extracted_contents, indent=2)}
"""
    }
