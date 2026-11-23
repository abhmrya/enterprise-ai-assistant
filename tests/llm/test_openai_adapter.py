
from types import SimpleNamespace
from unittest.mock import Mock

from enterprise_ai_assistant.config.settings import Settings
from enterprise_ai_assistant.llm.contracts import (
    GenerationConfig,
    LLMMessage,
    LLMRequest,
    MessageRole,
)
from enterprise_ai_assistant.llm.openai_adapter import OpenAIAdapter


def test_openai_adapter_converts_request_and_response() -> None:
    settings = Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
    )

    adapter = OpenAIAdapter(settings)

    fake_response = SimpleNamespace(
        output_text="MCP allows applications to connect with tools.",
        model="test-model",
        status="completed",
        usage=SimpleNamespace(
            input_tokens=20,
            output_tokens=12,
        ),
    )

    adapter._client.responses.create = Mock(
        return_value=fake_response
    )

    request = LLMRequest(
        messages=(
            LLMMessage(
                role=MessageRole.SYSTEM,
                content="You are an enterprise AI assistant.",
            ),
            LLMMessage(
                role=MessageRole.USER,
                content="Explain MCP.",
            ),
        ),
        config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=500,
        ),
    )

    result = adapter.generate(request)

    adapter._client.responses.create.assert_called_once_with(
        model="test-model",
        input=[
            {
                "role": "system",
                "content": "You are an enterprise AI assistant.",
            },
            {
                "role": "user",
                "content": "Explain MCP.",
            },
        ],
        temperature=0.2,
        max_output_tokens=500,
    )

    assert result.content == (
        "MCP allows applications to connect with tools."
    )
    assert result.model == "test-model"
    assert result.finish_reason == "completed"
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 12
