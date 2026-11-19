"""Unit tests for multi-provider LLM client (Gemini, OpenAI, Azure OpenAI)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.genai.llm_client import (
    LLMConfigError,
    _clean_json_content,
    call_chat,
    call_chat_json,
    get_client_and_model,
    get_model_name,
    get_provider_name,
    reset_client,
)


@pytest.fixture(autouse=True)
def clean_llm_env():
    """Clear LLM-related environment variables and reset client cache before each test."""
    reset_client()
    env_vars = [
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_API_BASE",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_API_BASE",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_VERSION",
    ]
    old_values = {var: os.environ.get(var) for var in env_vars}
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    yield

    reset_client()
    for var, val in old_values.items():
        if val is not None:
            os.environ[var] = val
        elif var in os.environ:
            del os.environ[var]


def test_clean_json_content():
    """Test stripping code blocks from raw LLM responses."""
    assert _clean_json_content(None) == ""
    assert _clean_json_content("") == ""
    assert _clean_json_content('{"key": "value"}') == '{"key": "value"}'
    assert _clean_json_content('```json\n{"key": "value"}\n```') == '{"key": "value"}'
    assert _clean_json_content('```\n{"key": "value"}\n```') == '{"key": "value"}'


def test_no_config_raises_llm_config_error():
    """When no LLM keys are present, LLMConfigError is raised."""
    with pytest.raises(LLMConfigError) as exc_info:
        get_client_and_model()
    assert "No LLM provider configured" in str(exc_info.value)


def test_gemini_provider_selected():
    """When GEMINI_API_KEY is present, Gemini provider and default model are selected."""
    os.environ["GEMINI_API_KEY"] = "fake-gemini-key"

    client, model = get_client_and_model()
    assert model == "gemini-3.5-flash-lite"
    assert get_provider_name() == "gemini"
    assert "generativelanguage.googleapis.com" in str(client.base_url)


def test_gemini_custom_model_selected():
    """Custom GEMINI_MODEL is respected."""
    os.environ["GEMINI_API_KEY"] = "fake-gemini-key"
    os.environ["GEMINI_MODEL"] = "gemini-1.5-pro"

    _, model = get_client_and_model()
    assert model == "gemini-1.5-pro"


def test_openai_provider_selected():
    """When OPENAI_API_KEY is present (and no GEMINI_API_KEY), OpenAI is selected."""
    os.environ["OPENAI_API_KEY"] = "sk-fake-openai-key"

    client, model = get_client_and_model()
    assert model == "gpt-4o-mini"
    assert get_provider_name() == "openai"


def test_azure_provider_selected():
    """When AZURE_OPENAI_* variables are set (and no Gemini/OpenAI key), Azure is selected."""
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://my-resource.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_KEY"] = "fake-azure-key"
    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o"

    client, model = get_client_and_model()
    assert model == "gpt-4o"
    assert get_provider_name() == "azure"


@patch("src.genai.llm_client.get_client_and_model")
def test_call_chat_json_mocked(mock_get_client_and_model):
    """Test call_chat_json invoking chat.completions.create in JSON mode."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"result": "ok"}'
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    mock_get_client_and_model.return_value = (mock_client, "test-model")

    res = call_chat_json("System prompt", "User prompt")
    assert res == '{"result": "ok"}'
    mock_client.chat.completions.create.assert_called_once_with(
        model="test-model",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"},
        ],
    )
