from __future__ import annotations

import hashlib
import json
import time

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

from src.config import (
    WEB_CACHE_DIR,
    WEB_CACHE_TTL,
    WEB_FETCH_TOP_K,
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
    - Rank search results using lightweight source-quality
      heuristics.
    - Fetch only the most useful sources.
    - Fall back to search snippets when fetching fails.
    - Cache research results for a short configurable TTL.
    - Collect structured web evidence.
    - Never crash the caller because one web source fails.
    """

    def __init__(
        self,
        max_results: int | None = None,
        max_workers: int | None = None,
    ):
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

        self.fetch_top_k = max(
            1,
            int(WEB_FETCH_TOP_K),
        )

        self.cache_ttl = max(
            0,
            int(WEB_CACHE_TTL),
        )

        self.cache_dir = Path(
            WEB_CACHE_DIR
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================
    # SEARCH
    # =========================================================

    def search(self, query):
        """
        Search the web.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        return self.search_engine.search(
            query.strip()
        )

    # =========================================================
    # CACHE
    # =========================================================

    @staticmethod
    def _normalize_query(query):
        """
        Normalize a query so equivalent searches share
        the same cache entry.
        """

        return " ".join(
            query.lower().strip().split()
        )

    def _cache_path(self, query):
        """
        Generate a deterministic cache filename.
        """

        normalized = self._normalize_query(
            query
        )

        query_hash = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        return (
            self.cache_dir
            / f"{query_hash}.json"
        )

    def _load_cache(self, query):
        """
        Load a valid cache entry.

        Returns None when:
        - no cache exists
        - cache is invalid
        - cache has expired
        """

        if self.cache_ttl <= 0:
            return None

        cache_path = self._cache_path(
            query
        )

        if not cache_path.exists():
            return None

        try:
            with cache_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                cached = json.load(file)

            cached_at = float(
                cached.get(
                    "cached_at",
                    0,
                )
            )

            if (
                time.time() - cached_at
                > self.cache_ttl
            ):
                return None

            result = cached.get(
                "result"
            )

            if not isinstance(
                result,
                dict,
            ):
                return None

            print(
                "⚡ Using cached web research."
            )

            return result

        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            return None

    def _save_cache(
        self,
        query,
        result,
    ):
        """
        Save a research result to disk.
        """

        if self.cache_ttl <= 0:
            return

        cache_path = self._cache_path(
            query
        )

        payload = {
            "cached_at": time.time(),
            "query": query,
            "result": result,
        }

        try:
            with cache_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    payload,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

        except OSError as exc:
            print(
                f"⚠️ Could not save web cache: "
                f"{exc}"
            )

    # =========================================================
    # SOURCE QUALITY
    # =========================================================

    @staticmethod
    def _source_quality_score(result):
        """
        Assign a lightweight quality score to a search result.

        This is intentionally heuristic rather than a hard
        allowlist so that useful sources are not accidentally
        discarded.
        """

        url = (
            result.get("url", "")
            .strip()
            .lower()
        )

        title = (
            result.get("title", "")
            .strip()
            .lower()
        )

        score = 0

        # Strong signals.
        high_quality_domains = [
            ".gov",
            ".gov.in",
            ".edu",
            ".ac.uk",
            "reuters.com",
            "apnews.com",
            "bbc.com",
            "nature.com",
            "who.int",
            "worldbank.org",
            "oecd.org",
            "weforum.org",
            "microsoft.com",
            "google.com",
            "ibm.com",
            "openai.com",
        ]

        if any(
            domain in url
            for domain in high_quality_domains
        ):
            score += 5

        # News-oriented domains.
        news_domains = [
            "nytimes.com",
            "theguardian.com",
            "cnn.com",
            "cnbc.com",
            "forbes.com",
            "techcrunch.com",
            "wired.com",
        ]

        if any(
            domain in url
            for domain in news_domains
        ):
            score += 3

        # Research / technical indicators.
        research_terms = [
            "research",
            "study",
            "report",
            "whitepaper",
            "documentation",
            "docs",
            "journal",
        ]

        if any(
            term in url or term in title
            for term in research_terms
        ):
            score += 2

        # Penalize obvious low-information pages.
        low_quality_terms = [
            "casino",
            "coupon",
            "advert",
            "sponsored",
        ]

        if any(
            term in url
            for term in low_quality_terms
        ):
            score -= 5

        # Prefer HTTPS.
        if url.startswith("https://"):
            score += 1

        return score

    def _rank_results(self, results):
        """
        Rank results by source quality while preserving
        search-engine relevance for equal scores.
        """

        ranked = []

        for index, result in enumerate(
            results
        ):
            item = dict(result)

            item["_quality_score"] = (
                self._source_quality_score(
                    item
                )
            )

            item["_search_position"] = index

            ranked.append(item)

        ranked.sort(
            key=lambda item: (
                -item["_quality_score"],
                item["_search_position"],
            )
        )

        return ranked

    # =========================================================
    # FETCHING
    # =========================================================

    def _fetch_result(
        self,
        index,
        result,
    ):
        """
        Fetch one web result.

        Falls back to the search snippet when the page
        cannot be fetched.
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

        quality_score = result.get(
            "_quality_score",
            0,
        )

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

        content = (
            page_text
            or snippet
        )

        if not content:
            return None

        return {
            "title": title,
            "url": url,
            "content": content,
            "fetched": bool(page_text),
            "quality_score": quality_score,
        }

    # =========================================================
    # EVIDENCE COLLECTION
    # =========================================================

    def collect_evidence(self, query):
        """
        Search, rank, fetch and structure web evidence.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        query = query.strip()

        # -----------------------------------------------------
        # Check cache first.
        # -----------------------------------------------------

        cached = self._load_cache(
            query
        )

        if cached is not None:
            return cached

        # -----------------------------------------------------
        # Search.
        # -----------------------------------------------------

        try:
            results = self.search(
                query
            )

        except Exception as exc:
            print(
                f"⚠️ Web search unavailable: "
                f"{exc}"
            )

            return {
                "evidence": [],
                "sources": [],
                "available": False,
                "error": str(exc),
                "cached": False,
            }

        if not results:
            result = {
                "evidence": [],
                "sources": [],
                "available": True,
                "error": None,
                "cached": False,
            }

            self._save_cache(
                query,
                result,
            )

            return result

        # -----------------------------------------------------
        # Rank sources.
        # -----------------------------------------------------

        ranked_results = self._rank_results(
            results
        )

        # -----------------------------------------------------
        # Fetch only top sources.
        # -----------------------------------------------------

        fetch_results = ranked_results[
            : self.fetch_top_k
        ]

        evidence = []

        worker_count = min(
            self.max_workers,
            len(fetch_results),
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
                    fetch_results,
                    start=1,
                )
            }

            for future in as_completed(
                futures
            ):
                index = futures[future]

                try:
                    result = future.result()

                    if result:
                        evidence.append(
                            result
                        )

                except Exception as exc:
                    print(
                        f"⚠️ Unexpected error "
                        f"for source {index}: "
                        f"{exc}"
                    )

        # -----------------------------------------------------
        # If fewer than fetch_top_k sources were successfully
        # fetched, use remaining search snippets as fallback.
        # -----------------------------------------------------

        existing_urls = {
            item.get("url")
            for item in evidence
        }

        for result in ranked_results:

            if len(evidence) >= self.fetch_top_k:
                break

            url = (
                result.get("url", "")
                .strip()
            )

            if not url:
                continue

            if url in existing_urls:
                continue

            snippet = (
                result.get("snippet", "")
                .strip()
            )

            if not snippet:
                continue

            evidence.append(
                {
                    "title": result.get(
                        "title",
                        "",
                    ),
                    "url": url,
                    "content": snippet,
                    "fetched": False,
                    "quality_score": result.get(
                        "_quality_score",
                        0,
                    ),
                }
            )

            existing_urls.add(
                url
            )

        # -----------------------------------------------------
        # Restore quality/search ordering.
        # -----------------------------------------------------

        result_positions = {
            item.get("url"): position
            for position, item in enumerate(
                ranked_results
            )
        }

        evidence.sort(
            key=lambda item: (
                -int(
                    item.get(
                        "quality_score",
                        0,
                    )
                ),
                result_positions.get(
                    item.get("url"),
                    999999,
                ),
            )
        )

        # -----------------------------------------------------
        # Build source metadata.
        # -----------------------------------------------------

        sources = []

        for item in evidence:
            sources.append(
                {
                    "title": item["title"],
                    "url": item["url"],
                    "type": "web",
                    "fetched": item[
                        "fetched"
                    ],
                    "quality_score": item.get(
                        "quality_score",
                        0,
                    ),
                }
            )

        result = {
            "evidence": evidence,
            "sources": sources,
            "available": True,
            "error": None,
            "cached": False,
        }

        self._save_cache(
            query,
            result,
        )

        return result

    # =========================================================
    # RESEARCH
    # =========================================================

    def research(self, query):
        """
        Collect web evidence without generating an LLM answer.

        LLM answer generation remains the responsibility
        of AssistantPipeline.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        query = query.strip()

        collected = (
            self.collect_evidence(
                query
            )
        )

        evidence = collected.get(
            "evidence",
            [],
        )

        sources = collected.get(
            "sources",
            [],
        )

        return {
            "question": query,
            "evidence": evidence,
            "sources": sources,
            "web_research": {
                "available": collected.get(
                    "available",
                    False,
                ),
                "sources_found": len(
                    evidence
                ),
                "error": collected.get(
                    "error"
                ),
                "cached": collected.get(
                    "cached",
                    False,
                ),
            },
        }