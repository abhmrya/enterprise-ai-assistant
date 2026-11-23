
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def main() -> None:
    # 1. System instructions
    system_message = SystemMessage(
        content="You are an Enterprise AI Assistant."
    )

    # 2. User request
    human_message = HumanMessage(
        content="Find employee 101's department."
    )

    # 3. AI tool-call request (illustrative)
    ai_tool_request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_employee",
                "args": {"employee_id": 101},
                "id": "call_001",
                "type": "tool_call",
            }
        ],
    )

    # 4. Tool result
    tool_result = ToolMessage(
        content='{"employee_id": 101, "department": "Engineering"}',
        tool_call_id="call_001",
    )

    # 5. Final AI response
    ai_response = AIMessage(
        content="Employee 101 works in Engineering."
    )

    conversation = [
        system_message,
        human_message,
        ai_tool_request,
        tool_result,
        ai_response,
    ]

    for index, message in enumerate(conversation, start=1):
        print(f"\nMessage {index}")
        print("Type:", type(message).__name__)
        print("Content:", message.content)
        print("Additional kwargs:", message.additional_kwargs)

        if isinstance(message, AIMessage):
            print("Tool calls:", message.tool_calls)

        if isinstance(message, ToolMessage):
            print("Tool call ID:", message.tool_call_id)


if __name__ == "__main__":
    main()
