class GeneralPromptBuilder:
    """
    Builds prompts for questions that can be answered
    using general LLM knowledge.
    """

    def build_prompt(self, query):
        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        return f"""
You are a helpful AI assistant.

Answer the user's question using your general
knowledge and reasoning.

Instructions:
- Answer the question directly.
- Do not mention document retrieval.
- Do not pretend that you searched the web.
- Do not fabricate sources or citations.
- If you are uncertain, clearly say so.
- Keep the answer concise and useful.
- Use a few sentences unless more detail is necessary.

QUESTION:
{query}

ANSWER:
"""