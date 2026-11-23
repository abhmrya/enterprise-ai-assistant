
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser


def main() -> None:
    # Simulate an AI model response
    ai_response = AIMessage(
        content="Employee 101 belongs to Engineering."
    )

    # Create the parser
    parser = StrOutputParser()

    # Parse AIMessage into a Python string
    result = parser.invoke(ai_response)

    print("\n========== PARSED OUTPUT ==========")
    print(result)

    print("\n========== PYTHON TYPE ==========")
    print(type(result))


if __name__ == "__main__":
    main()
