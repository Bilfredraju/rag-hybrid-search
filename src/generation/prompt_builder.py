class PromptBuilder:
    """
    Builds prompts for the LLM using retrieved documents.
    """

    def build_prompt(self, query, documents):

        context = ""

        for i, doc in enumerate(documents, start=1):

            # Limit each chunk to 500 characters
            chunk = doc["document"][:900]

            context += (
                f"Document {i}\n"
                f"Source: {doc['metadata']['source']}\n"
                f"Page: {doc['metadata']['page']}\n"
                f"Content:\n{chunk}\n\n"
            )

        prompt = f"""

You are an Enterprise AI Assistant answering inside a chat widget.

Answer the user's question using ONLY the information provided below.

Instructions:
- Write a single short paragraph of 3-4 sentences. Do NOT use bullet points.
- Start directly with the answer. No preamble like "Based on the documents...".
- If the context mentions any of the following, you MUST include them: eligibility rules or restrictions (e.g. probation/notice period), approval process, accumulation/encashment rules, and any exceptions.
- Include specific numbers (days, dates, timeframes) exactly as stated in the context — never say "a certain number" or "maximum number" if the context gives an actual figure.
- Do not repeat document titles or placeholders such as "Name of the Company".
- Keep the total answer under 90 words, but never omit a rule or exception to save space — trim wording, not content.
- Use only the provided context.
- If the answer is not found, reply exactly:
  "I couldn't find that information in the provided documents."

================ CONTEXT ================

{context}

=========================================

QUESTION:
{query}

ANSWER (3-4 sentences, no bullets, under 90 words, include all rules/numbers/exceptions from context):
"""

        return prompt