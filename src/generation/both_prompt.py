class BothPromptBuilder:
    """
    Builds a synthesis prompt using both document evidence
    and current web evidence.
    """

    def build_prompt(self, query, document_results, web_results):
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        if not document_results and not web_results:
            raise ValueError(
                "At least one document or web result is required."
            )

        evidence_parts = []

        if document_results:
            evidence_parts.append(
                "================ DOCUMENT EVIDENCE ================\n"
            )

            for index, result in enumerate(
                document_results,
                start=1,
            ):
                metadata = result.get("metadata", {})

                source = metadata.get(
                    "source",
                    "Unknown source",
                )

                page = metadata.get(
                    "page",
                    "Unknown page",
                )

                content = result.get(
                    "document",
                    "",
                )[:900]

                evidence_parts.append(
                    f"""
Document Evidence {index}
Source: {source}
Page: {page}
Content:
{content}
"""
                )

        if web_results:
            evidence_parts.append(
                "\n================ WEB EVIDENCE ================\n"
            )

            for index, result in enumerate(
                web_results,
                start=1,
            ):
                title = result.get(
                    "title",
                    "Unknown title",
                )

                url = result.get(
                    "url",
                    "",
                )

                content = result.get(
                    "content",
                    result.get("snippet", ""),
                )[:6000]

                evidence_parts.append(
                    f"""
Web Evidence {index}
Title: {title}
URL: {url}
Content:
{content}
"""
                )

        evidence = "\n".join(evidence_parts)

        return f"""
You are an AI research assistant.

Answer the user's question using ONLY the supplied
document evidence and web evidence.

STRICT GROUNDING RULES:

- Do not use outside knowledge.
- Do not invent facts.
- Do not fabricate sources or URLs.
- Clearly distinguish information from the uploaded
  documents from current web information.
- If the document evidence and web evidence disagree,
  explicitly state that they disagree.
- If one evidence source is insufficient, say so.
- Do not create citation markers such as [1], [2],
  【1】, 【1†L1-L2】, or similar.
- Do not create a bibliography.
- Do not mention the retrieval process.
- Answer the user's exact question.
- Keep the answer concise.
- Prefer 3-6 sentences unless more detail is necessary.

The application will return source metadata separately.
Therefore, do NOT write URLs or source numbers inside
the answer.

{evidence}

=======================================================

QUESTION:
{query}

ANSWER:
"""
