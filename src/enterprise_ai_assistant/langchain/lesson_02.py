
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from enterprise_ai_assistant.config.settings import Settings


def main() -> None:
    # Load existing project configuration.
    settings = Settings.from_environment()

    # Create the LangChain Chat Model.
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        max_tokens=500,
        timeout=30,
        max_retries=2,
    )

    # Prepare structured messages.
    messages = [
        SystemMessage(
            content="You are an Enterprise AI Assistant."
        ),
        HumanMessage(
            content="Explain what LangChain is in simple terms."
        ),
    ]

    # Execute the model synchronously.
    response = model.invoke(messages)

    # Display the generated content.
    print("\n========== AI RESPONSE ==========\n")
    print(response.content)

    # Inspect the response object.
    print("\n========== RESPONSE DETAILS ==========")
    print("Message type:", type(response).__name__)
    print("Model:", response.response_metadata.get("model_name"))
    print("Usage:", response.usage_metadata)


if __name__ == "__main__":
    main()
