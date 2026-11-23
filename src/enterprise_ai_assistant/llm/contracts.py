
from dataclasses import dataclass
from enum import StrEnum


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class LLMMessage:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.2
    max_output_tokens: int = 1000

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0 and 2")

        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


@dataclass(frozen=True)
class LLMRequest:
    messages: tuple[LLMMessage, ...]
    config: GenerationConfig = GenerationConfig()

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("At least one message is required")


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    finish_reason: str
    usage: TokenUsage
