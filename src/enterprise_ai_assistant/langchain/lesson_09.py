
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage


# 1. Define our expected data structure
class EmployeeInfo(BaseModel):
    employee_id: int = Field(description="Unique employee ID")
    department: str = Field(description="Employee department")
    status: Literal["active", "inactive"] = Field(
        description="Employment status"
    )


def main() -> None:
    # 2. Simulate a JSON response from an AI model
    ai_response = AIMessage(
        content="""
        {
            "employee_id": 101,
            "department": "Engineering",
            "status": "active"
        }
        """
    )

    # 3. Create the parser
    parser = PydanticOutputParser(
        pydantic_object=EmployeeInfo
    )

    # 4. Parse and validate
    result = parser.invoke(ai_response)

    print("\n========== PARSED EMPLOYEE ==========")
    print(result)

    print("\n========== FIELD ACCESS ==========")
    print("Employee ID:", result.employee_id)
    print("Department:", result.department)
    print("Status:", result.status)

    print("\n========== PYTHON TYPE ==========")
    print(type(result))


if __name__ == "__main__":
    main()
