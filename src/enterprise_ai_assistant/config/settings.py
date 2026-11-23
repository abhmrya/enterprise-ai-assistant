
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"

    openai_api_key: str | None = None
    openai_model: str = "gpt-5.5"

    @classmethod
    def from_environment(cls) -> "Settings":
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not gemini_api_key or gemini_api_key == (
            "PASTE_YOUR_GEMINI_API_KEY_HERE"
        ):
            raise RuntimeError(
                "A valid GEMINI_API_KEY must be configured in .env"
            )

        return cls(
            gemini_api_key=gemini_api_key,
            gemini_model=os.getenv(
                "GEMINI_MODEL",
                "gemini-3.8-flash",
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
        )
