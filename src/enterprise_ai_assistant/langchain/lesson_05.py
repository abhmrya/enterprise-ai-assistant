
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


def main() -> None:
    # 1. Create a prompt with a dynamic history placeholder
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an Enterprise AI Assistant. "
                "Answer accurately and use conversation history "
                "when relevant.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "{question}",
            ),
        ]
    )

    # 2. Prepare previous conversation messages
    chat_history = [
        HumanMessage(content="Find employee 101's department."),
        AIMessage(content="Employee 101 belongs to Engineering."),
    ]

    # 3. Supply history and the current question
    prompt_value = prompt_template.invoke(
        {
            "chat_history": chat_history,
            "question": "Which department does employee 101 belong to?",
        }
    )

    # 4. Inspect the final messages
    print("\n========== GENERATED CONVERSATION ==========")

    for message in prompt_value.to_messages():
        print(f"\n{type(message).__name__}: {message.content}")


if __name__ == "__main__":
    main()
