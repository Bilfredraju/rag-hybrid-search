class PromptBuilder:
    """
    Builds grounded prompts for the LLM using retrieved documents.

    The LLM must answer only from the supplied evidence and
    should focus specifically on the user's question.
    """

    def build_prompt(self, query, documents):

        if not documents:
            raise ValueError(
                "At least one document is required "
                "to build the prompt."
            )

        context_parts = []

        for i, doc in enumerate(
            documents,
            start=1,
        ):
            metadata = doc.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
                "Unknown source",
            )

            page = metadata.get(
                "page",
                "Unknown page",
            )

            chunk = doc.get(
                "document",
                "",
            )

            # Keep the context reasonably sized.
            chunk = chunk[:900]

            context_parts.append(
                f"""
Evidence {i}
Source: {source}
Page: {page}
Content:
{chunk}
"""
            )

        context = "\n".join(
            context_parts
        )

        prompt = f"""
You are an Enterprise AI Assistant.

Answer the user's question using ONLY the evidence
provided below.

STRICT GROUNDING RULES:
- Do not use outside knowledge.
- Do not invent, assume, or infer unsupported facts.
- Every factual claim in your answer must be supported
  by the provided evidence.
- Answer specifically what the user asked.
- Do not add unrelated facts simply because they appear
  in the evidence.
- If the evidence contains the exact answer, state it
  clearly and directly.
- Preserve important names, dates, numbers, and terms
  exactly as they appear in the evidence.
- If the evidence is insufficient to answer the question,
  reply exactly:
  "I couldn't find that information in the provided documents."
- Do not mention the retrieval process.
- Do not say "According to the documents" or
  "Based on the context".
- Do not fabricate citations or sources.
- Keep the answer concise.
- Use a single paragraph.
- Prefer 1-3 sentences unless more detail is necessary
  to answer the question completely.

================ EVIDENCE ================

{context}

===========================================

QUESTION:
{query}

ANSWER:
"""

        return prompt