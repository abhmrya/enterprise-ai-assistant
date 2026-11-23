
from langchain_core.prompts import ChatPromptTemplate


def main() -> None:
    # 1. Define a reusable chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an Enterprise AI Assistant.

                Answer employee questions using the provided context.
                If the context does not contain the answer,
                clearly say that you don't know.
                """,
            ),
            (
                "human",
                """
                Context:
                {context}

                Employee Question:
                {question}
                """,
            ),
        ]
    )

    # 2. Supply runtime values
    prompt_value = prompt_template.invoke(
        {
            "context": "Employee 101 belongs to the Engineering department.",
            "question": "Which department does employee 101 work in?",
        }
    )

    # 3. Inspect the generated messages
    print("\n========== GENERATED MESSAGES ==========")

    for message in prompt_value.to_messages():
        print(f"\nType: {type(message).__name__}")
        print(f"Content: {message.content}")

    # 4. Inspect the complete prompt as text
    print("\n========== COMPLETE PROMPT ==========")
    print(prompt_value.to_string())


if __name__ == "__main__":
    main()
