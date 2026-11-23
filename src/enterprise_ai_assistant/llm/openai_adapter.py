
from openai import OpenAI

from enterprise_ai_assistant.config.settings import Settings
from enterprise_ai_assistant.llm.contracts import (
    LLMRequest,
    LLMResponse,
    TokenUsage,
)


class OpenAIAdapter:
    """
    Adapter responsible for communicating with the OpenAI API.

    It translates our internal LLM contracts into OpenAI API requests
    and converts API responses back into our internal response format.
    """

    def __init__(
        self,
        settings: Settings,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._settings = settings

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        messages = [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in request.messages
        ]

        response = self._client.responses.create(
            model=self._settings.openai_model,
            input=messages,
            temperature=request.config.temperature,
            max_output_tokens=request.config.max_output_tokens,
        )

        usage = response.usage

        return LLMResponse(
            content=response.output_text,
            model=response.model,
            finish_reason=response.status or "unknown",
            usage=TokenUsage(
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
            ),
        )
