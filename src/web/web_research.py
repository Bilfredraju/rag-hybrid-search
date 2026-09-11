from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import (
    WEB_MAX_RESULTS,
    WEB_MAX_WORKERS,
)
from src.web.web_fetcher import WebFetcher
from src.web.web_search import WebSearch


class WebResearch:
    """
    Web research layer.

    Responsibilities:
    - Search the web.
    - Fetch readable page content when possible.
    - Fall back to search snippets when fetching fails.
    - Collect structured web evidence.
    - Never crash the caller because one web source fails.

    LLM answer generation is intentionally handled by
    AssistantPipeline rather than this class.
    """

    def __init__(
        self,
        max_results: int | None = None,
        max_workers: int | None = None,
    ):
        # Use configured values unless explicitly overridden.
        self.search_engine = WebSearch(
            max_results=(
                max_results
                if max_results is not None
                else WEB_MAX_RESULTS
            )
        )

        self.fetcher = WebFetcher()

        self.max_workers = max(
            1,
            int(
                max_workers
                if max_workers is not None
                else WEB_MAX_WORKERS
            ),
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(self, query):
        """
        Search the web and return normalized search results.

        Raises:
            ValueError: if the query is empty.
            RuntimeError: if the search engine fails.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        return self.search_engine.search(
            query.strip()
        )

    # ========================================================
    # FETCH INDIVIDUAL RESULT
    # ========================================================

    def _fetch_result(self, index, result):
        """
        Fetch a single search result.

        If the webpage cannot be fetched, the search
        snippet is used as a fallback.

        Returns:
            dict | None
        """

        title = result.get(
            "title",
            "",
        ).strip()

        url = result.get(
            "url",
            "",
        ).strip()

        snippet = result.get(
            "snippet",
            "",
        ).strip()

        if not url:
            return None

        page_text = ""

        try:
            page_text = self.fetcher.fetch(
                url
            )

        except Exception as exc:
            print(
                f"⚠️ Could not fetch source "
                f"{index}: {exc}"
            )

        # ----------------------------------------------------
        # Fallback to search-engine snippet
        # ----------------------------------------------------

        content = page_text or snippet

        if not content:
            return None

        return {
            "title": title,
            "url": url,
            "content": content,
            "fetched": bool(page_text),
        }

    # ========================================================
    # COLLECT EVIDENCE
    # ========================================================

    def collect_evidence(self, query):
        """
        Search the web and collect usable evidence.

        Web search failures are converted into structured
        responses instead of propagating exceptions.

        Individual page-fetch failures also do not stop
        the remaining sources from being processed.

        Returns:
            {
                "evidence": [...],
                "sources": [...],
                "available": bool,
                "error": str | None
            }
        """

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        try:
            results = self.search(
                query
            )

        except Exception as exc:
            print(
                f"⚠️ Web search unavailable: {exc}"
            )

            return {
                "evidence": [],
                "sources": [],
                "available": False,
                "error": str(exc),
            }

        # ----------------------------------------------------
        # No search results
        # ----------------------------------------------------

        if not results:
            return {
                "evidence": [],
                "sources": [],
                "available": True,
                "error": None,
            }

        evidence = []

        # ----------------------------------------------------
        # Fetch pages concurrently
        # ----------------------------------------------------

        worker_count = min(
            self.max_workers,
            len(results),
        )

        with ThreadPoolExecutor(
            max_workers=worker_count
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

            for future in as_completed(
                futures
            ):
                index = futures[future]

                try:
                    result = future.result()

                    if result:
                        completed.append(
                            result
                        )

                except Exception as exc:
                    print(
                        f"⚠️ Unexpected error "
                        f"for source {index}: "
                        f"{exc}"
                    )

        # ----------------------------------------------------
        # Preserve search-engine ranking
        # ----------------------------------------------------

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

        evidence.extend(
            completed
        )

        # ----------------------------------------------------
        # Build source metadata
        # ----------------------------------------------------

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

    # ========================================================
    # SIMPLE RESEARCH API
    # ========================================================

    def research(self, query):
        """
        Collect web evidence without generating an LLM answer.

        This method is intentionally lightweight.

        The AssistantPipeline is responsible for taking
        this evidence and generating the final answer.

        Returns:
            {
                "question": str,
                "evidence": [...],
                "sources": [...],
                "web_research": {
                    "available": bool,
                    "sources_found": int,
                    "error": str | None
                }
            }
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        query = query.strip()

        collected = self.collect_evidence(
            query
        )

        evidence = collected[
            "evidence"
        ]

        sources = collected[
            "sources"
        ]

        return {
            "question": query,
            "evidence": evidence,
            "sources": sources,
            "web_research": {
                "available": collected[
                    "available"
                ],
                "sources_found": len(
                    evidence
                ),
                "error": collected[
                    "error"
                ],
            },
        }