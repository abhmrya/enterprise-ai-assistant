
from enterprise_ai_assistant.llm.contracts import (
    LLMRequest,
    LLMResponse,
    LLMMessage,
    MessageRole,
    TokenUsage,
)
from enterprise_ai_assistant.llm.service import LLMService


class FakeLLMProvider:
    """In-memory provider used for isolated service testing."""

    def __init__(self) -> None:
        self.received_request: LLMRequest | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.received_request = request

        return LLMResponse(
            content="This is a fake LLM response.",
            model="fake-model",
            finish_reason="completed",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=8,
            ),
        )


def test_llm_service_delegates_request_to_provider() -> None:
    provider = FakeLLMProvider()
    service = LLMService(provider=provider)

    request = LLMRequest(
        messages=(
            LLMMessage(
                role=MessageRole.USER,
                content="Explain dependency injection.",
            ),
        )
    )

    response = service.generate_response(request)

    assert provider.received_request is request
    assert response.content == "This is a fake LLM response."
    assert response.model == "fake-model"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 8
