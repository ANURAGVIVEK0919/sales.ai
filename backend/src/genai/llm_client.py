"""Multi-provider LLM client supporting Google Gemini, OpenAI, and Azure OpenAI.

Shared by POST /forecast/explain and POST /chat.

Supports 3 providers in priority order:
1. Google Gemini API (Free tier available at https://aistudio.google.com/apikey):
   - GEMINI_API_KEY (required for Gemini)
   - GEMINI_MODEL (optional, defaults to 'gemini-2.0-flash')
   - GEMINI_API_BASE (optional, defaults to 'https://generativelanguage.googleapis.com/v1beta/openai/')

2. Standard OpenAI:
   - OPENAI_API_KEY (required for direct OpenAI)
   - OPENAI_MODEL (optional, defaults to 'gpt-4o-mini')
   - OPENAI_API_BASE (optional, defaults to 'https://api.openai.com/v1')

3. Azure OpenAI:
   - AZURE_OPENAI_ENDPOINT
   - AZURE_OPENAI_API_KEY
   - AZURE_OPENAI_DEPLOYMENT_NAME
   - AZURE_OPENAI_API_VERSION (optional, defaults to '2024-06-01')

Reads credentials from environment variables (loaded from .env locally via
python-dotenv) -- never hardcode a key here.
"""

import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessage

# Load .env from backend directory and project root
_curr = Path(__file__).resolve().parent
load_dotenv(_curr.parent.parent / ".env")
load_dotenv(_curr.parent.parent.parent / ".env")

_client = None
_model_name = None
_provider = None


class LLMConfigError(RuntimeError):
    """Raised when required LLM environment variables are missing."""


def reset_client() -> None:
    """Reset cached client (useful in tests or when env vars change)."""
    global _client, _model_name, _provider
    _client = None
    _model_name = None
    _provider = None


def get_client_and_model() -> Tuple[Any, str]:
    """Return a cached (client, model_name) tuple, or raise LLMConfigError."""
    global _client, _model_name, _provider

    if _client is not None and _model_name is not None:
        return _client, _model_name

    # Check 1: Google Gemini (OpenAI-compatible endpoint)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        base_url = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
        _model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        _client = OpenAI(
            api_key=gemini_key,
            base_url=base_url,
        )
        _provider = "gemini"
        return _client, _model_name

    # Check 2: Standard OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        _model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        _client = OpenAI(
            api_key=openai_key,
            base_url=base_url,
        )
        _provider = "openai"
        return _client, _model_name

    # Check 3: Azure OpenAI
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

    if azure_endpoint and azure_key and azure_deployment:
        _model_name = azure_deployment
        _client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=azure_version,
        )
        _provider = "azure"
        return _client, _model_name

    # Unconfigured
    raise LLMConfigError(
        "No LLM provider configured. Set GEMINI_API_KEY (for Google Gemini), "
        "OPENAI_API_KEY (for OpenAI), or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + "
        "AZURE_OPENAI_DEPLOYMENT_NAME (for Azure OpenAI) in backend/.env"
    )


def get_client() -> Any:
    """Return a cached client, or raise LLMConfigError if unconfigured."""
    client, _ = get_client_and_model()
    return client


def get_model_name() -> str:
    """Return the active model/deployment name, or raise LLMConfigError if unconfigured."""
    _, model = get_client_and_model()
    return model


def get_provider_name() -> str:
    """Return the active provider name ('gemini', 'openai', 'azure')."""
    get_client_and_model()
    return _provider or "unknown"


def _clean_json_content(content: Optional[str]) -> str:
    """Strip markdown code fences if an LLM wraps its JSON response in ```json ... ```."""
    if not content:
        return ""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    return text


def call_chat_json(system_prompt: str, user_message: str) -> str:
    """Call the configured LLM provider in JSON mode.

    Returns the raw response content (a JSON string, per response_format).
    Raises LLMConfigError if credentials are missing; propagates the
    underlying openai SDK's exceptions on network/API failures so the caller
    can translate those into a clean HTTP error.
    """
    client, model = get_client_and_model()

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    content = response.choices[0].message.content or ""
    return _clean_json_content(content)


def call_chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    json_mode: bool = False,
) -> ChatCompletionMessage:
    """General-purpose chat completion call, optionally with tool definitions
    and/or JSON-mode output. Used by the /chat tool-calling flow.

    Raises LLMConfigError if credentials are missing; propagates the
    underlying openai SDK's exceptions on network/API failures.
    """
    client, model = get_client_and_model()

    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    if json_mode and message.content:
        message.content = _clean_json_content(message.content)
    return message


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick manual test for the LLM client.")
    parser.add_argument("--name", default="World", help="Name to greet in the test prompt")
    args = parser.parse_args()

    try:
        reply = call_chat_json(
            'You are a helpful assistant. Respond with JSON in this exact shape: {"message": "..."}.',
            f"Say hello to {args.name} in one short sentence.",
        )
        print(f"Provider: {get_provider_name()}, Model: {get_model_name()}")
        print(reply)
    except LLMConfigError as err:
        print(f"Configuration error: {err}")
