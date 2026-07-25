import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LLM:
    """
    Groq LLM Wrapper
    """

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env"
            )

        self.client = Groq(api_key=api_key)

        # Recommended Groq model
        self.model_name = "llama-3.3-70b-versatile"

        print(f"✅ Groq Client Loaded ({self.model_name})")

    def generate(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=220,
        )

        return response.choices[0].message.content