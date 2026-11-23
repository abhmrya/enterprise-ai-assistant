
import asyncio

from langchain_core.runnables import RunnableLambda


# Normal Python function.
def process_question(question: str) -> str:
    cleaned_question = question.strip()

    return (
        f"Enterprise Assistant received your question: "
        f"{cleaned_question}"
    )


async def main() -> None:
    # Wrap the Python function as a LangChain Runnable.
    question_runnable = RunnableLambda(process_question)

    # 1. INVOKE
    print("\n========== INVOKE ==========")

    result = question_runnable.invoke(
        "What is employee leave policy?"
    )

    print(result)

    # 2. BATCH
    print("\n========== BATCH ==========")

    questions = [
        "What is Python?",
        "What is LangChain?",
        "What is MCP?",
    ]

    results = question_runnable.batch(questions)

    for result in results:
        print(result)

    # 3. STREAM
    print("\n========== STREAM ==========")

    for chunk in question_runnable.stream(
        "Explain Enterprise AI Assistant"
    ):
        print(chunk)

    # 4. AINVOKE
    print("\n========== AINVOKE ==========")

    async_result = await question_runnable.ainvoke(
        "Explain asynchronous execution"
    )

    print(async_result)


if __name__ == "__main__":
    asyncio.run(main())
