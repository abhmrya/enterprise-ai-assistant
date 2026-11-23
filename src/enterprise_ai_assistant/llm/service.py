
from typing import Protocol

from enterprise_ai_assistant.llm.contracts import (
    LLMRequest,
    LLMResponse,
)


class LLMProvider(Protocol):
    """
    Contract that every LLM provider must follow.
    """

    def generate(self, request: LLMRequest) -> LLMResponse:
        ...


class LLMService:
    """
    Application service responsible for executing LLM requests.

    It depends on an abstraction, not a concrete provider.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def generate_response(self, request: LLMRequest) -> LLMResponse:
        return self._provider.generate(request)
