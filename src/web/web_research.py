from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.generation.llm import LLM
from src.web.web_fetcher import WebFetcher
from src.web.web_search import WebSearch


class WebResearch:
    """
    Web research layer.

    Responsibilities:
    - Search the web.
    - Fetch readable page content when possible.
    - Fall back to search snippets when fetching fails.
    - Never crash the caller because one web source fails.
    """

    def __init__(
        self,
        max_results=5,
        max_workers=5,
    ):
        self.search_engine = WebSearch(
            max_results=max_results
        )
        self.fetcher = WebFetcher()
        self.llm = LLM()
        self.max_workers = max(1, int(max_workers))

    def search(self, query):
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        return self.search_engine.search(query.strip())

    def _fetch_result(self, index, result):
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        snippet = result.get("snippet", "").strip()

        if not url:
            return None

        page_text = ""

        try:
            page_text = self.fetcher.fetch(url)
        except Exception as exc:
            print(
                f"⚠️ Could not fetch source {index}: {exc}"
            )

        # If fetching fails, use the search-engine snippet.
        content = page_text or snippet

        if not content:
            return None

        return {
            "title": title,
            "url": url,
            "content": content,
            "fetched": bool(page_text),
        }

    def collect_evidence(self, query):
        """
        Search and collect web evidence.

        Web failures are converted into a structured result
        instead of being allowed to crash the assistant.
        """

        try:
            results = self.search(query)
        except Exception as exc:
            print(f"⚠️ Web search unavailable: {exc}")

            return {
                "evidence": [],
                "sources": [],
                "available": False,
                "error": str(exc),
            }

        if not results:
            return {
                "evidence": [],
                "sources": [],
                "available": True,
                "error": None,
            }

        evidence = []

        with ThreadPoolExecutor(
            max_workers=min(
                self.max_workers,
                len(results),
            )
        ) as executor:

            futures = {
                executor.submit(
                    self._fetch_result,
                    index,
                    result,
                ): index
                for index, result in enumerate(
                    results,
                    start=1,
                )
            }

            completed = []

            for future in as_completed(futures):
                index = futures[future]

                try:
                    result = future.result()

                    if result:
                        completed.append(result)

                except Exception as exc:
                    print(
                        f"⚠️ Unexpected error for source "
                        f"{index}: {exc}"
                    )

        # Preserve the original search ranking.
        result_positions = {
            item.get("url"): position
            for position, item in enumerate(results)
        }

        completed.sort(
            key=lambda item: result_positions.get(
                item.get("url"),
                999999,
            )
        )

        evidence.extend(completed)

        sources = [
            {
                "title": item["title"],
                "url": item["url"],
                "type": "web",
                "fetched": item["fetched"],
            }
            for item in evidence
        ]

        return {
            "evidence": evidence,
            "sources": sources,
            "available": True,
            "error": None,
        }

    def research(self, query):
        """
        Perform web research and generate an answer.

        If web search is unavailable, return a structured
        failure instead of raising an exception.
        """

        if not query or not query.strip():
            raise ValueError("query must not be empty")

        query = query.strip()

        collected = self.collect_evidence(query)

        evidence = collected["evidence"]
        sources = collected["sources"]

        if not collected["available"]:
            return {
                "question": query,
                "answer": (
                    "Web research is temporarily unavailable. "
                    "I couldn't retrieve reliable current "
                    "web information for this question."
                ),
                "sources": [],
                "web_research": {
                    "available": False,
                    "sources_found": 0,
                    "error": collected["error"],
                },
            }

        if not evidence:
            return {
                "question": query,
                "answer": (
                    "I couldn't find reliable web sources "
                    "for this question."
                ),
                "sources": [],
                "web_research": {
                    "available": True,
                    "sources_found": 0,
                    "error": None,
                },
            }

        context_parts = []

        for index, item in enumerate(
            evidence,
            start=1,
        ):
            context_parts.append(
                f"""
Web Evidence {index}
Title: {item["title"]}
URL: {item["url"]}
Content:
{item["content"][:6000]}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are a web research assistant.

Answer the user's question using ONLY the
web evidence provided below.

STRICT RULES:

- Use only the supplied web evidence.
- Do not invent facts.
- Do not use outside knowledge.
- Do not fabricate sources or URLs.
- Clearly mention uncertainty when evidence
  is insufficient or sources disagree.
- Do not create citation markers such as [1],
  [2], 【1】, 【1†L1-L2】, or similar.
- Do not create a bibliography.
- Do not mention the retrieval process.
- Do not claim that you personally visited
  a website.
- Answer the user's exact question.
- Keep the answer concise.
- Prefer 2-5 sentences unless more detail
  is necessary.

The application will return source metadata
separately.

Therefore, DO NOT write URLs, source numbers,
or citation markers inside the answer.

================ WEB EVIDENCE ================

{context}

===============================================

QUESTION:
{query}

ANSWER:
"""

        try:
            answer = self.llm.generate(prompt)

        except Exception as exc:
            print(
                f"⚠️ Web answer generation failed: {exc}"
            )

            return {
                "question": query,
                "answer": (
                    "I found relevant web sources, but "
                    "the AI answer generation service is "
                    "temporarily unavailable."
                ),
                "sources": sources,
                "web_research": {
                    "available": True,
                    "sources_found": len(evidence),
                    "error": str(exc),
                },
            }

        return {
            "question": query,
            "answer": answer,
            "sources": sources,
            "web_research": {
                "available": True,
                "sources_found": len(evidence),
                "error": None,
            },
        }