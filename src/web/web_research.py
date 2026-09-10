from __future__ import annotations

from src.generation.llm import LLM
from src.web.web_fetcher import WebFetcher
from src.web.web_search import WebSearch


class WebResearch:
    """
    Web research pipeline.

    Flow:
        Query
          ↓
        Search
          ↓
        Fetch pages
          ↓
        Extract readable text
          ↓
        LLM synthesis
          ↓
        Answer + verified source metadata
    """

    def __init__(self, max_results=5):
        self.search_engine = WebSearch(max_results=max_results)
        self.fetcher = WebFetcher()
        self.llm = LLM()

    def search(self, query):
        return self.search_engine.search(query)

    def _build_evidence(self, results):
        evidence_parts = []
        verified_sources = []

        for index, result in enumerate(results, start=1):
            title = result.get("title", "").strip()
            url = result.get("url", "").strip()
            snippet = result.get("snippet", "").strip()

            if not url:
                continue

            page_text = ""

            try:
                page_text = self.fetcher.fetch(url)
            except Exception as exc:
                print(
                    f"⚠️ Could not fetch source {index}: "
                    f"{exc}"
                )

            # Prefer actual page text.
            # Fall back to search snippet if fetching fails.
            content = page_text or snippet

            if not content:
                continue

            evidence_parts.append(
                f"""
Web Evidence {len(evidence_parts) + 1}
Title: {title}
URL: {url}
Content:
{content}
"""
            )

            verified_sources.append(
                {
                    "title": title,
                    "url": url,
                    "type": "web",
                    "fetched": bool(page_text),
                }
            )

        return "\n".join(evidence_parts), verified_sources

    def research(self, query):
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        query = query.strip()

        results = self.search(query)

        if not results:
            return {
                "question": query,
                "answer": (
                    "I couldn't find reliable web sources "
                    "for this question."
                ),
                "sources": [],
            }

        context, sources = self._build_evidence(results)

        if not context:
            return {
                "question": query,
                "answer": (
                    "I couldn't retrieve reliable content "
                    "from the web sources found."
                ),
                "sources": sources,
            }

        prompt = f"""
You are a web research assistant.

Answer the user's question using ONLY the web evidence
provided below.

STRICT RULES:

- Use only the supplied web evidence.
- Do not invent facts.
- Do not use outside knowledge.
- Do not fabricate sources.
- Do not fabricate URLs.
- Do not create citation markers such as [1], [2],
  【1】, 【1†L1-L2】, or similar.
- Do not create a bibliography.
- Do not mention the retrieval process.
- Do not claim that you personally visited or verified
  a website.
- If sources disagree, clearly mention the disagreement.
- If the evidence is insufficient, say so.
- Answer the user's exact question.
- Keep the answer concise.
- Prefer 2-5 sentences unless more detail is necessary.

The application will attach the actual source metadata
separately. Therefore, DO NOT write source numbers,
citation markers, URLs, or references inside your answer.

================ WEB EVIDENCE ================

{context}

===============================================

QUESTION:
{query}

ANSWER:
"""

        answer = self.llm.generate(prompt)

        return {
            "question": query,
            "answer": answer,
            "sources": sources,
        }