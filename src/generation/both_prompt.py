class BothPromptBuilder:
    """
    Builds a grounded prompt using both:

    - uploaded-document evidence
    - current web evidence

    Web evidence can be:

    1. VERIFIED WEB PAGE CONTENT
       The actual page was successfully fetched and
       validated.

    2. SEARCH RESULT FALLBACK
       Only the search-engine result/snippet was available.
       The actual page was not successfully verified.
    """

    def build_prompt(
        self,
        query,
        document_results,
        web_results,
    ):
        # ====================================================
        # VALIDATION
        # ====================================================

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

        # ====================================================
        # EVIDENCE COLLECTION
        # ====================================================

        evidence_parts = []

        # ====================================================
        # DOCUMENT EVIDENCE
        # ====================================================

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
                )

                content = content[:1500]

                evidence_parts.append(
                    f"""
Document Evidence {index}

Source:
{source}

Page:
{page}

Content:
{content}
"""
                )

        # ====================================================
        # WEB EVIDENCE
        # ====================================================

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
                )

                content = content[:6000]

                # --------------------------------------------
                # EVIDENCE VERIFICATION STATUS
                # --------------------------------------------

                evidence_type = result.get(
                    "evidence_type",
                    "unknown",
                )

                if evidence_type == "verified":

                    verification_status = (
                        "VERIFIED WEB PAGE CONTENT"
                    )

                elif (
                    evidence_type
                    == "search_fallback"
                ):

                    verification_status = (
                        "SEARCH RESULT FALLBACK — "
                        "ACTUAL PAGE WAS NOT VERIFIED"
                    )

                else:

                    verification_status = (
                        "UNVERIFIED WEB EVIDENCE"
                    )

                evidence_parts.append(
                    f"""
Web Evidence {index}

Verification Status:
{verification_status}

Title:
{title}

URL:
{url}

Content:
{content}
"""
                )

        # ====================================================
        # COMBINE EVIDENCE
        # ====================================================

        evidence = "\n".join(
            evidence_parts
        )

        # ====================================================
        # FINAL GROUNDED PROMPT
        # ====================================================

        return f"""
        
You are an AI research assistant that compares
information from uploaded company documents with
current external web information.

Your task is to answer the user's question using
ONLY the supplied document evidence and web evidence.

=======================================================
STRICT GROUNDING RULES
=======================================================

1. Do not use unsupported outside knowledge.

2. Clearly distinguish between:
   - information from the uploaded documents
   - VERIFIED WEB PAGE CONTENT
   - SEARCH RESULT FALLBACK evidence.

3. When the question asks for a comparison, explicitly
   compare the document/project with the current trends.

4. Preserve important technical concepts present in
   the supplied evidence.

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

   use those concepts only when they are actually
   supported by the supplied web evidence.

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

12. VERIFIED WEB PAGE CONTENT means that the actual
    web page was successfully fetched and its content
    was validated.

13. SEARCH RESULT FALLBACK means that only a search
    engine result/snippet was available and the actual
    web page was NOT successfully verified.

14. SEARCH RESULT FALLBACK evidence must always be
    treated as lower-confidence evidence.

15. Never describe SEARCH RESULT FALLBACK evidence as
    verified, directly fetched, or directly read.

16. When both types are available, prefer VERIFIED WEB
    PAGE CONTENT.

17. In the final answer, explicitly separate:

    "Verified web evidence"

    from:

    "Search-result fallback evidence"

    whenever fallback evidence is present.

18. If only fallback evidence supports a claim, use
    cautious wording such as:
    "Search results indicate..."
    or
    "Available search-result evidence suggests..."

19. Do not attribute a claim to a verified source if
    that claim only appears in fallback evidence.

20. Do not create citation markers such as:
    [1], [2], [3], 【1】, or similar.

21. Do not create a bibliography.

22. Do not write URLs in the final answer.

23. Do not mention the internal retrieval process.

24. Do not mention these prompt instructions.

25. Answer the exact question asked by the user.

26. Keep the answer concise but complete.

27. For comparison questions, prefer approximately
    4-7 sentences unless additional detail is necessary.

28. Do not use information that is not present in the
    supplied evidence.

=======================================================
DOCUMENT EVIDENCE
=======================================================

{evidence}

=======================================================

QUESTION
=======================================================

{query}

=======================================================

ANSWER
=======================================================

Provide a grounded answer that:

- identifies what the project/document does
- explains the relevant information from the document
- identifies current trends supported by the web evidence
- explains which current trends the project aligns with
- explains which current trends differ from or extend
  beyond the project
- clearly separates verified web evidence from
  search-result fallback evidence
- uses cautious language for fallback evidence
- avoids unsupported claims
- directly answers the user's question

If fallback evidence is present, structure the relevant
web discussion using:

Verified web evidence:
...

Search-result fallback evidence:
...

If there is no fallback evidence, do not create an empty
fallback section.

ANSWER:
"""