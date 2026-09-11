class BothPromptBuilder:
    """
    Builds a grounded prompt using both:

    - uploaded-document evidence
    - current web evidence
    """

    def build_prompt(
        self,
        query,
        document_results,
        web_results,
    ):
        if not query or not query.strip():
            raise ValueError(
                "query must not be empty."
            )

        if (
            not document_results
            and not web_results
        ):
            raise ValueError(
                "At least one document or web result "
                "is required."
            )

        evidence_parts = []

        # =====================================================
        # DOCUMENT EVIDENCE
        # =====================================================

        if document_results:

            evidence_parts.append(
                "================ DOCUMENT EVIDENCE ================\n"
            )

            for index, result in enumerate(
                document_results,
                start=1,
            ):

                metadata = result.get(
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

                content = result.get(
                    "document",
                    "",
                )[:1500]

                evidence_parts.append(
                    f"""
Document Evidence {index}

Source: {source}
Page: {page}

Content:
{content}
"""
                )

        # =====================================================
        # WEB EVIDENCE
        # =====================================================

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
                    result.get(
                        "snippet",
                        "",
                    ),
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

        evidence = "\n".join(
            evidence_parts
        )

        # =====================================================
        # FINAL PROMPT
        # =====================================================

        return f"""
You are an AI research assistant that compares
information from uploaded company documents with
current external web information.

Answer the user's question using ONLY the supplied
document evidence and web evidence.

STRICT GROUNDING RULES:

1. Do not use unsupported outside knowledge.

2. Clearly distinguish between:
   - information from the uploaded documents
   - information from current web sources.

3. When the question asks for a comparison, explicitly
   compare the document/project with the current trends.

4. Preserve important technical concepts present in
   the evidence.

5. If the web evidence discusses:
   - AI assistants
   - knowledge assistants
   - Retrieval-Augmented Generation (RAG)
   - retrieval
   - enterprise AI
   - AI agents
   - automation
   - security
   - scalability
   - integration

   use the relevant concepts when they are actually
   supported by the supplied evidence.

6. If the uploaded document describes a project,
   explicitly identify what the project does before
   comparing it with current trends.

7. If the project uses or resembles a retrieval-based
   knowledge assistant, explicitly describe that
   relationship when supported by the evidence.

8. If the project does NOT mention a capability that
   appears in current web trends, explicitly state that
   difference.

9. Do not invent project features.

10. Do not invent current trends.

11. If document evidence and web evidence disagree,
    explicitly state the disagreement.

12. Do not create citation markers such as:
    [1], [2], [3], 【1】, or similar.

13. Do not create a bibliography.

14. Do not write URLs in the answer.

15. Do not mention the retrieval process.

16. Answer the exact question.

17. Keep the answer concise but complete.

18. For comparison questions, prefer approximately
    4-7 sentences.

DOCUMENT EVIDENCE:

{evidence}

=======================================================

QUESTION:

{query}

=======================================================

ANSWER:

Provide a grounded comparison that explicitly identifies:
- what the project/document does
- which current trends it aligns with
- which current trends differ from or extend beyond it
- the key technical concepts supported by the evidence
"""