
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from enterprise_ai_assistant.config.settings import Settings


class EmployeeInfo(BaseModel):
    employee_id: int = Field(description="Unique employee ID")
    department: str = Field(description="Employee department")
    status: Literal["active", "inactive"] = Field(
        description="Employment status"
    )


def main() -> None:
    settings = Settings.from_environment()

    # 1. Create the output parser
    parser = PydanticOutputParser(
        pydantic_object=EmployeeInfo
    )

    # 2. Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an Enterprise AI Assistant.

                Return employee information using the requested format.

                {format_instructions}

                Only use information provided in the employee context.
                Do not invent employee details.
                """,
            ),
            (
                "human",
                """
                Employee context:
                {employee_context}

                Question:
                {question}
                """,
            ),
        ]
    )

    # 3. Create the Gemini model
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
        max_tokens=500,
        timeout=30,
        max_retries=2,
    )

    # 4. Connect components using LCEL
    chain = prompt | model | parser

    # 5. Execute the pipeline
    result = chain.invoke(
        {
            "format_instructions": parser.get_format_instructions(),
            "employee_context": (
                "Employee 101 belongs to Engineering. "
                "Employment status is active."
            ),
            "question": (
                "Find employee 101's department and status."
            ),
        }
    )

    # 6. Display structured result
    print("\n========== STRUCTURED EMPLOYEE DATA ==========")
    print(result)

    print("\n========== INDIVIDUAL FIELDS ==========")
    print("Employee ID:", result.employee_id)
    print("Department:", result.department)
    print("Status:", result.status)

    print("\n========== RESULT TYPE ==========")
    print(type(result))


if __name__ == "__main__":
    main()
