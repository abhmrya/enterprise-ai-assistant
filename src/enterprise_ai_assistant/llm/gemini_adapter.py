
from google import genai
from google.genai import errors, types

from enterprise_ai_assistant.config.settings import Settings
from enterprise_ai_assistant.llm.contracts import (
    LLMRequest,
    LLMResponse,
    MessageRole,
    TokenUsage,
)
from enterprise_ai_assistant.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMUnavailableError,
)


class GeminiAdapter:
    """Gemini implementation of our provider-independent LLM interface."""

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key is required")

        self._settings = settings

        self._client = genai.Client(
            api_key=settings.gemini_api_key,
        )

    @staticmethod
    def _translate_api_error(exc: errors.APIError) -> None:
        """Translate Gemini SDK errors into application-level exceptions."""

        status_code = exc.code
        message = str(exc)

        if status_code in (401, 403):
            raise LLMAuthenticationError(
                "Gemini authentication or permission failed."
            ) from exc

        if status_code == 429:
            raise LLMRateLimitError(
                "Gemini API rate limit or quota exceeded."
            ) from exc

        if status_code is not None and status_code >= 500:
            raise LLMUnavailableError(
                "Gemini service is temporarily unavailable."
            ) from exc

        raise LLMProviderError(
            f"Gemini API request failed: {message}"
        ) from exc

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the Gemini API."""

        system_messages = []
        conversation_messages = []

        # Convert application messages into Gemini message format.
        for message in request.messages:
            if message.role == MessageRole.SYSTEM:
                system_messages.append(message.content)
                continue

            role = (
                "model"
                if message.role == MessageRole.ASSISTANT
                else "user"
            )

            conversation_messages.append(
                types.Content(
                    role=role,
                    parts=[
                        types.Part(text=message.content),
                    ],
                )
            )

        if not conversation_messages:
            raise ValueError(
                "At least one user or assistant message is required"
            )

        # Combine multiple system messages.
        system_instruction = (
            "\n\n".join(system_messages)
            if system_messages
            else None
        )

        # Configure generation.
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=request.config.max_output_tokens,
        )

        # Call Gemini and translate provider-specific errors.
        try:
            response = self._client.models.generate_content(
                model=self._settings.gemini_model,
                contents=conversation_messages,
                config=config,
            )
        except errors.APIError as exc:
            self._translate_api_error(exc)

        # Extract token usage.
        usage = response.usage_metadata

        # Extract finish reason.
        candidates = response.candidates or []

        finish_reason = (
            str(candidates[0].finish_reason)
            if candidates
            else "unknown"
        )

        # Return provider-independent response.
        return LLMResponse(
            content=response.text or "",
            model=self._settings.gemini_model,
            finish_reason=finish_reason,
            usage=TokenUsage(
                input_tokens=(
                    usage.prompt_token_count
                    if usage
                    else 0
                ),
                output_tokens=(
                    usage.candidates_token_count
                    if usage
                    else 0
                ),
            ),
        )
