
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


def main() -> None:
    # 1. Create a reusable prompt
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are {assistant_name}.
                Your department is {department}.
                Answer questions accurately.
                """,
            ),
            MessagesPlaceholder(
                variable_name="chat_history",
                optional=True,
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    # 2. Pre-fill variables that usually remain constant
    enterprise_prompt = prompt_template.partial(
        assistant_name="Enterprise AI Assistant",
        department="Engineering",
    )

    # 3. Invoke without providing chat history
    prompt_value = enterprise_prompt.invoke(
        {
            "question": "What does our Engineering department do?"
        }
    )

    # 4. Inspect generated messages
    print("\n========== GENERATED PROMPT ==========")

    for message in prompt_value.to_messages():
        print(f"\n{type(message).__name__}:")
        print(message.content)

    # 5. Inspect remaining required variables
    print("\n========== REMAINING INPUT VARIABLES ==========")
    print(enterprise_prompt.input_variables)


if __name__ == "__main__":
    main()
