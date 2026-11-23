
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage


def main() -> None:
    # Simulate JSON returned by an AI model
    ai_response = AIMessage(
        content="""
        {
            "employee_id": 101,
            "department": "Engineering",
            "status": "active"
        }
        """
    )

    # Create JSON parser
    parser = JsonOutputParser()

    # Convert AIMessage into Python data
    result = parser.invoke(ai_response)

    print("\n========== PARSED JSON ==========")
    print(result)

    print("\n========== PYTHON TYPE ==========")
    print(type(result))

    print("\n========== ACCESS INDIVIDUAL FIELDS ==========")
    print("Employee ID:", result["employee_id"])
    print("Department:", result["department"])
    print("Status:", result["status"])


if __name__ == "__main__":
    main()
