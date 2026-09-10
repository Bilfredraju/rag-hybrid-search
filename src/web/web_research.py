from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.generation.llm import LLM
from src.web.web_fetcher import WebFetcher
from src.web.web_search import WebSearch


class WebResearch:
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

        self.max_workers = max(
            1,
            int(max_workers),
        )

    def search(self, query):
        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        return self.search_engine.search(
            query.strip()
        )

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
                f"⚠️ Could not fetch source "
                f"{index}: {exc}"
            )

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
        results = self.search(query)

        if not results:
            return {
                "evidence": [],
                "sources": [],
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
                try:
                    result = future.result()

                    if result:
                        completed.append(result)

                except Exception as exc:
                    index = futures[future]

                    print(
                        f"⚠️ Unexpected error "
                        f"for source {index}: {exc}"
                    )

        # Preserve search-engine ranking order.
        result_positions = {
            item.get("url"): position
            for position, item in enumerate(
                results
            )
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
        }

    def research(self, query):
        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        query = query.strip()

        collected = self.collect_evidence(query)

        evidence = collected["evidence"]
        sources = collected["sources"]

        if not evidence:
            return {
                "question": query,
                "answer": (
                    "I couldn't find reliable "
                    "web sources for this question."
                ),
                "sources": [],
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

        answer = self.llm.generate(prompt)

        return {
            "question": query,
            "answer": answer,
            "sources": sources,
        }