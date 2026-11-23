
import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str

    @classmethod
    def from_environment(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-5.5")

        if not api_key or api_key == "your_actual_api_key_here":
            raise RuntimeError(
                "A valid OPENAI_API_KEY must be configured in .env"
            )

        return cls(
            openai_api_key=api_key,
            openai_model=model,
        )
