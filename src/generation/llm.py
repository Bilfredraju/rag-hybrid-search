import os

from dotenv import load_dotenv
from groq import Groq

from src.config import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")


class LLM:
    """Groq-based LLM generation layer."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in the .env file."
            )

        self.model_name = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )

        self.client = Groq(api_key=api_key)

        print(
            f"Groq Client Loaded ({self.model_name})"
        )

    def generate(self, prompt):
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_tokens=500,
            reasoning_effort="low",
            include_reasoning=False,
        )

        content = response.choices[0].message.content

        if not content or not content.strip():
            raise RuntimeError(
                "Groq returned an empty answer."
            )

        return content.strip()