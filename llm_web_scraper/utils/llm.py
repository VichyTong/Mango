import os
from typing import Optional
from dotenv import load_dotenv
from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings
import litellm

litellm.drop_params = True

load_dotenv()

PROVIDER_ENDPOINTS = {
    'openai': {
        'api_key_env': 'OPENAI_API_KEY',
        'model_prefixes': ['gpt-'],
        'default_model': 'gpt-4.1',
        'litellm_prefix': 'openai/',
    },
    'gemini': {
        'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/',
        'api_key_env': 'GOOGLE_API_KEY',
        'model_prefixes': ['gemini-'],
        'default_model': 'gemini-2.5-flash',
        'litellm_prefix': 'gemini/',
    },
    'qwen': {
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key_env': 'DASHSCOPE_API_KEY',
        'model_prefixes': ['qwen-', 'qwen2-', 'qwen3-'],
        'default_model': 'qwen3-4b',
        'litellm_prefix': 'dashscope/',
    },
    'forge': {
        'base_url': 'https://api.forge.tensorblock.co/v1',
        'api_key_env': 'FORGE_API_KEY',
        'model_prefixes': ['tensorblock/gpt-'],
        'default_model': 'tensorblock/gpt-4.1',
        'litellm_prefix': 'tensorblock/',
    },
    'local': {
        'base_url': os.getenv('LOCAL_LLM_BASE_URL', 'http://localhost:8000/v1'),
        'api_key': 'EMPTY',
        'litellm_prefix': 'openai/',
    },
}

PRESET_PARAMETERS = {
    "qwen3": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0
    },
    "gpt-4.1": {
        "temperature": 0.7
    }
}


def _detect_provider_from_model(model_name: str) -> str:
    model_lower = model_name.lower()
    for provider, config in PROVIDER_ENDPOINTS.items():
        prefixes = config.get('model_prefixes', [])
        if any(prefix in model_lower for prefix in prefixes):
            return provider
    return 'openai'


def get_litellm_model(
    model_name: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None
) -> LitellmModel:
    if not provider:
        provider = _detect_provider_from_model(model_name)
        if not provider:
            raise ValueError("Provider must be specified if model_name is not given.")

    provider_config = PROVIDER_ENDPOINTS.get(provider)

    litellm_prefix = provider_config.get('litellm_prefix', '')
    litellm_model_name = f"{litellm_prefix}{model_name}" if litellm_prefix else model_name

    if not api_key:
        if 'api_key' in provider_config:
            api_key = provider_config['api_key']
        else:
            api_key_env = provider_config.get('api_key_env')
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found. Please set {api_key_env} environment variable.")

    if provider == 'local':
        base_url = os.getenv('LOCAL_LLM_BASE_URL', 'http://localhost:8000/v1')
    else:
        base_url = provider_config.get('base_url', None)

    return LitellmModel(model=litellm_model_name, api_key=api_key, base_url=base_url)


def create_model_settings(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None
) -> ModelSettings:
    model_settings_dict = {}

    preset_params = {}
    if model_name:
        for preset_key, params in PRESET_PARAMETERS.items():
            if model_name.lower().startswith(preset_key.lower()):
                preset_params = params
                break

    if temperature is not None:
        model_settings_dict["temperature"] = temperature
    elif "temperature" in preset_params and preset_params["temperature"] is not None:
        model_settings_dict["temperature"] = preset_params["temperature"]

    if max_tokens is not None:
        model_settings_dict["max_tokens"] = max_tokens

    if top_p is not None:
        model_settings_dict["top_p"] = top_p
    elif "top_p" in preset_params and preset_params["top_p"] is not None:
        model_settings_dict["top_p"] = preset_params["top_p"]

    if top_k is not None:
        model_settings_dict["top_k"] = top_k
    elif "top_k" in preset_params and preset_params["top_k"] is not None:
        model_settings_dict["top_k"] = preset_params["top_k"]

    if min_p is not None:
        model_settings_dict["min_p"] = min_p
    elif "min_p" in preset_params and preset_params["min_p"] is not None:
        model_settings_dict["min_p"] = preset_params["min_p"]

    if timeout is None:
        timeout = 600
    model_settings_dict["timeout"] = timeout

    if model_name and "qwen" in model_name.lower():
        model_settings_dict["extra_args"] = {"enable_thinking": False}

    return ModelSettings(**model_settings_dict)


def count_tokens(raw_responses: list) -> int:
    tokens_used = 0
    for resp in raw_responses:
        tokens_used += resp.usage.total_tokens
    return tokens_used
