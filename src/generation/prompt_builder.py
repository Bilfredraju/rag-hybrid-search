class PromptBuilder:
    """
    Builds prompts for the LLM using retrieved documents.
    """

    def build_prompt(self, query, documents):

        context = ""

        for i, doc in enumerate(documents, start=1):

            # Limit each chunk to 500 characters
            chunk = doc["document"][:500]

            context += (
                f"Document {i}\n"
                f"Source: {doc['metadata']['source']}\n"
                f"Page: {doc['metadata']['page']}\n"
                f"Content:\n{chunk}\n\n"
            )

        prompt = f"""
You are an Enterprise AI Assistant.

Answer the user's question using ONLY the information provided below.


Instructions:
- Start directly with the answer.
- Do not repeat document titles or placeholders such as "Name of the Company".
- Summarize only the relevant information.
- Use 5–8 bullet points.
- Keep the answer between 100 and 200 words.
- Use only the provided context.
- If the answer is not found, reply exactly:
  "I couldn't find that information in the provided documents."
================ CONTEXT ================

{context}

=========================================

QUESTION:
{query}

ANSWER:
"""

        return prompt