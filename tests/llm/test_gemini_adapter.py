
from types import SimpleNamespace
from unittest.mock import Mock

from enterprise_ai_assistant.config.settings import Settings
from enterprise_ai_assistant.llm.contracts import (
    GenerationConfig,
    LLMMessage,
    LLMRequest,
    MessageRole,
)
from enterprise_ai_assistant.llm.gemini_adapter import GeminiAdapter


def test_gemini_adapter_converts_request_and_response() -> None:
    settings = Settings(
        gemini_api_key="test-api-key",
        gemini_model="gemini-3.5-flash-lite",
    )

    adapter = GeminiAdapter(settings=settings)

    fake_response = SimpleNamespace(
        text="LLM response",
        usage_metadata=SimpleNamespace(
            prompt_token_count=20,
            candidates_token_count=10,
        ),
        candidates=[
            SimpleNamespace(finish_reason="STOP"),
        ],
    )

    adapter._client = Mock()
    adapter._client.models.generate_content.return_value = fake_response

    request = LLMRequest(
        messages=(
            LLMMessage(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant.",
            ),
            LLMMessage(
                role=MessageRole.USER,
                content="Explain Python.",
            ),
        ),
        config=GenerationConfig(max_output_tokens=200),
    )

    response = adapter.generate(request)

    adapter._client.models.generate_content.assert_called_once()

    call_args = adapter._client.models.generate_content.call_args

    assert call_args.kwargs["model"] == "gemini-3.5-flash-lite"

    sent_messages = call_args.kwargs["contents"]

    assert len(sent_messages) == 1
    assert sent_messages[0].role == "user"
    assert sent_messages[0].parts[0].text == "Explain Python."

    assert (
        call_args.kwargs["config"].system_instruction
        == "You are a helpful assistant."
    )

    assert call_args.kwargs["config"].max_output_tokens == 200

    assert response.content == "LLM response"
    assert response.model == "gemini-3.5-flash-lite"
    assert response.finish_reason == "STOP"
    assert response.usage.input_tokens == 20
    assert response.usage.output_tokens == 10



def test_gemini_adapter_converts_assistant_message_to_model_role() -> None:
    settings = Settings(
        gemini_api_key="test-api-key",
        gemini_model="gemini-3.5-flash-lite",
    )

    adapter = GeminiAdapter(settings=settings)

    adapter._client = Mock()
    adapter._client.models.generate_content.return_value = SimpleNamespace(
        text="Second response",
        usage_metadata=None,
        candidates=[],
    )

    request = LLMRequest(
        messages=(
            LLMMessage(
                role=MessageRole.USER,
                content="Hello",
            ),
            LLMMessage(
                role=MessageRole.ASSISTANT,
                content="Hi there!",
            ),
            LLMMessage(
                role=MessageRole.USER,
                content="How are you?",
            ),
        )
    )

    response = adapter.generate(request)

    sent_messages = (
        adapter._client.models.generate_content.call_args.kwargs["contents"]
    )

    assert [message.role for message in sent_messages] == [
        "user",
        "model",
        "user",
    ]

    assert sent_messages[1].parts[0].text == "Hi there!"
    assert response.content == "Second response"
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0
    assert response.finish_reason == "unknown"


def test_gemini_adapter_handles_empty_response_text() -> None:
    settings = Settings(
        gemini_api_key="test-api-key",
        gemini_model="gemini-3.5-flash-lite",
    )

    adapter = GeminiAdapter(settings=settings)

    adapter._client = Mock()
    adapter._client.models.generate_content.return_value = SimpleNamespace(
        text=None,
        usage_metadata=None,
        candidates=[],
    )

    request = LLMRequest(
        messages=(
            LLMMessage(
                role=MessageRole.USER,
                content="Generate something",
            ),
        )
    )

    response = adapter.generate(request)

    assert response.content == ""
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0
