
from enterprise_ai_assistant.config.settings import Settings
from enterprise_ai_assistant.llm.contracts import (
    GenerationConfig,
    LLMMessage,
    LLMRequest,
    MessageRole,
)
from enterprise_ai_assistant.llm.gemini_adapter import GeminiAdapter
from enterprise_ai_assistant.llm.service import LLMService


def main() -> None:
    settings = Settings.from_environment()

    provider = GeminiAdapter(settings=settings)
    llm_service = LLMService(provider=provider)

    request = LLMRequest(
        messages=(
            LLMMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are an enterprise AI engineering assistant. "
                    "Give technically accurate and practical explanations."
                ),
            ),
            LLMMessage(
                role=MessageRole.USER,
                content="Explain how an LLM generates a response.",
            ),
        ),
        config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=500,
        ),
    )

    response = llm_service.generate_response(request)

    print("\n========== LLM RESPONSE ==========\n")
    print(response.content)

    print("\n========== TOKEN USAGE ==========")
    print("Provider: Gemini")
    print(f"Model: {response.model}")
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")
    print(f"Status: {response.finish_reason}")


if __name__ == "__main__":
    main()
