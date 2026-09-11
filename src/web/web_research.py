from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.config import (
    WEB_CACHE_DIR,
    WEB_CACHE_TTL,
    WEB_FETCH_TOP_K,
    WEB_FETCH_TIMEOUT,
    WEB_MAX_CHARS,
    WEB_MAX_RESULTS,
    WEB_MAX_WORKERS,
)

from src.web.web_fetcher import WebFetcher
from src.web.web_search import WebSearch


class WebResearch:
    """
    Web research pipeline.

    Responsibilities:
    1. Search the web.
    2. Classify the query type.
    3. Rank sources using quality + query relevance.
    4. Fetch the best sources.
    5. Return grounded web evidence.
    6. Cache research results for a configurable TTL.
    """

    def __init__(
        self,
        max_results: int | None = None,
        max_workers: int | None = None,
        fetch_timeout: int | None = None,
        max_chars: int | None = None,
        fetch_top_k: int | None = None,
        cache_ttl: int | None = None,
        cache_dir: Path | str | None = None,
    ):
        self.max_results = (
            max_results
            if max_results is not None
            else WEB_MAX_RESULTS
        )

        self.max_workers = (
            max_workers
            if max_workers is not None
            else WEB_MAX_WORKERS
        )

        self.fetch_top_k = (
            fetch_top_k
            if fetch_top_k is not None
            else WEB_FETCH_TOP_K
        )

        self.cache_ttl = (
            cache_ttl
            if cache_ttl is not None
            else WEB_CACHE_TTL
        )

        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else WEB_CACHE_DIR
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.searcher = WebSearch(
            max_results=self.max_results
        )

        self.fetcher = WebFetcher(
            timeout=(
                fetch_timeout
                if fetch_timeout is not None
                else WEB_FETCH_TIMEOUT
            ),
            max_chars=(
                max_chars
                if max_chars is not None
                else WEB_MAX_CHARS
            ),
        )

    # =========================================================
    # QUERY TYPE DETECTION
    # =========================================================

    @staticmethod
    def _detect_query_type(query: str) -> str:
        """
        Detect the broad category of a web query.

        Categories:
        - finance
        - news
        - government
        - hr
        - research
        - technology
        - general

        Domain-specific intent is checked before generic
        freshness words such as "latest".
        """

        normalized = " ".join(
            query.lower().strip().split()
        )

        # -----------------------------------------------------
        # FINANCE
        # -----------------------------------------------------

        finance_terms = [
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "crypto",
            "cryptocurrency",
            "stock price",
            "share price",
            "market price",
            "stock market",
            "financial market",
            "finance",
            "financial",
            "nasdaq",
            "nyse",
            "exchange rate",
            "currency",
            "gold price",
            "oil price",
        ]

        if any(
            term in normalized
            for term in finance_terms
        ):
            return "finance"

        # -----------------------------------------------------
        # GOVERNMENT
        # -----------------------------------------------------

        government_terms = [
            "government",
            "government regulation",
            "government regulations",
            "law",
            "laws",
            "legislation",
            "legal",
            "regulation",
            "regulations",
            "official notification",
            "government policy",
            "public policy",
            "ministry",
            "parliament",
        ]

        if any(
            term in normalized
            for term in government_terms
        ):
            return "government"

        # -----------------------------------------------------
        # HR
        # -----------------------------------------------------

        hr_terms = [
            "employee",
            "employees",
            "human resources",
            "hr",
            "leave management",
            "employee leave",
            "annual leave",
            "paid leave",
            "vacation",
            "absence management",
            "workplace",
            "workforce",
            "employee benefits",
            "employee policy",
            "hr trends",
            "hr management",
        ]

        if any(
            term in normalized
            for term in hr_terms
        ):
            return "hr"

        # -----------------------------------------------------
        # RESEARCH
        # -----------------------------------------------------

        research_terms = [
            "scientific research",
            "research paper",
            "research study",
            "academic research",
            "scientific study",
            "journal",
            "peer reviewed",
            "peer-reviewed",
            "academic paper",
            "scientific paper",
            "research findings",
        ]

        if any(
            term in normalized
            for term in research_terms
        ):
            return "research"

        # -----------------------------------------------------
        # NEWS
        # -----------------------------------------------------
        # IMPORTANT:
        # Do not use "latest" alone here.
        #
        # Otherwise:
        # "latest government regulations"
        # "latest employee leave trends"
        # "latest scientific research"
        #
        # would all become NEWS.

        news_terms = [
            "latest news",
            "breaking news",
            "recent news",
            "news update",
            "news updates",
            "current events",
            "headlines",
            "what happened",
            "recent developments",
            "latest developments",
            "breaking",
            "news",
        ]

        # Explicit AI/technology news queries.
        ai_news_terms = [
            "artificial intelligence news",
            "ai news",
            "latest ai news",
            "latest artificial intelligence news",
            "generative ai news",
            "machine learning news",
            "technology news",
            "tech news",
        ]

        if any(
            term in normalized
            for term in ai_news_terms
        ):
            return "news"

        if any(
            term in normalized
            for term in news_terms
        ):
            return "news"

        # -----------------------------------------------------
        # TECHNOLOGY
        # -----------------------------------------------------

        technology_terms = [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "generative ai",
            "large language model",
            "llm",
            "rag",
            "retrieval augmented generation",
            "ai assistant",
            "ai assistants",
            "technology",
            "software",
            "cloud computing",
            "robotics",
            "automation",
        ]

        if any(
            term in normalized
            for term in technology_terms
        ):
            return "technology"

        return "general"

    # =========================================================
    # BASE SOURCE QUALITY
    # =========================================================

    @staticmethod
    def _source_quality_score(result: dict) -> int:
        """
        Calculate baseline quality for a web source.
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

        text = f"{title} {url}"

        score = 0

        # -----------------------------------------------------
        # HIGH-QUALITY / AUTHORITATIVE SOURCES
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # NEWS / INDUSTRY SOURCES
        # -----------------------------------------------------

        news_domains = [
            "nytimes.com",
            "theguardian.com",
            "cnn.com",
            "cnbc.com",
            "forbes.com",
            "techcrunch.com",
            "wired.com",
            "apnews.com",
            "bbc.com",
        ]

        if any(
            domain in url
            for domain in news_domains
        ):
            score += 3

        # -----------------------------------------------------
        # RESEARCH SIGNAL
        # -----------------------------------------------------

        research_terms = [
            "research",
            "journal",
            "study",
            "paper",
            "scientific",
            "academic",
        ]

        if any(
            term in text
            for term in research_terms
        ):
            score += 2

        # -----------------------------------------------------
        # HTTPS
        # -----------------------------------------------------

        if url.startswith("https://"):
            score += 1

        # -----------------------------------------------------
        # LOW QUALITY
        # -----------------------------------------------------

        low_quality_terms = [
            "casino",
            "coupon",
            "advert",
            "advertisement",
            "sponsored",
        ]

        if any(
            term in text
            for term in low_quality_terms
        ):
            score -= 5

        return score

    # =========================================================
    # QUERY RELEVANCE SCORE
    # =========================================================

    @classmethod
    def _query_relevance_score(
        cls,
        query_type,
        result,
    ) -> int:

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

        text = f"{title} {url}"

        score = 0

        # -----------------------------------------------------
        # FINANCE
        # -----------------------------------------------------

        if query_type == "finance":

            finance_domains = [
                "reuters.com",
                "bloomberg.com",
                "cnbc.com",
                "wsj.com",
                "coindesk.com",
                "cointelegraph.com",
                "nasdaq.com",
                "nyse.com",
                "investing.com",
                "yahoo.com",
            ]

            if any(
                domain in url
                for domain in finance_domains
            ):
                score += 4

            finance_keywords = [
                "bitcoin",
                "btc",
                "ethereum",
                "eth",
                "crypto",
                "cryptocurrency",
                "market",
                "stock",
                "share",
                "price",
                "finance",
                "financial",
            ]

            keyword_matches = sum(
                1
                for keyword in finance_keywords
                if keyword in text
            )

            score += min(
                keyword_matches,
                4,
            )

        # -----------------------------------------------------
        # NEWS
        # -----------------------------------------------------

        elif query_type == "news":

            news_domains = [
                "reuters.com",
                "apnews.com",
                "bbc.com",
                "theguardian.com",
                "nytimes.com",
                "cnn.com",
                "cnbc.com",
                "forbes.com",
            ]

            if any(
                domain in url
                for domain in news_domains
            ):
                score += 4

            news_keywords = [
                "news",
                "latest",
                "breaking",
                "headline",
                "update",
                "developments",
                "artificial intelligence",
                "ai",
            ]

            keyword_matches = sum(
                1
                for keyword in news_keywords
                if keyword in text
            )

            score += min(
                keyword_matches,
                4,
            )

        # -----------------------------------------------------
        # TECHNOLOGY
        # -----------------------------------------------------

        elif query_type == "technology":

            technology_domains = [
                "techcrunch.com",
                "wired.com",
                "arstechnica.com",
                "microsoft.com",
                "google.com",
                "openai.com",
                "ibm.com",
                "mit.edu",
                "stanford.edu",
            ]

            if any(
                domain in url
                for domain in technology_domains
            ):
                score += 4

            technology_keywords = [
                "ai",
                "artificial intelligence",
                "machine learning",
                "deep learning",
                "rag",
                "retrieval",
                "llm",
                "large language model",
                "assistant",
                "generative",
                "agentic",
                "technology",
            ]

            keyword_matches = sum(
                1
                for keyword in technology_keywords
                if keyword in text
            )

            score += min(
                keyword_matches,
                4,
            )

        # -----------------------------------------------------
        # HR
        # -----------------------------------------------------

        elif query_type == "hr":

            hr_domains = [
                "shrm.org",
                "gartner.com",
                "mckinsey.com",
                "deloitte.com",
                "pwc.com",
                "ey.com",
                "oecd.org",
                "weforum.org",
                "nfp.com",
            ]

            if any(
                domain in url
                for domain in hr_domains
            ):
                score += 4

            hr_keywords = [
                "employee",
                "employees",
                "leave",
                "absence",
                "absences",
                "time off",
                "paid leave",
                "annual leave",
                "vacation",
                "workplace",
                "hr",
                "human resources",
                "workforce",
                "benefits",
                "leave management",
                "employee management",
            ]

            keyword_matches = sum(
                1
                for keyword in hr_keywords
                if keyword in text
            )

            score += min(
                keyword_matches,
                4,
            )

        # -----------------------------------------------------
        # GOVERNMENT
        # -----------------------------------------------------

        elif query_type == "government":

            government_domains = [
                ".gov",
                ".gov.in",
                ".gov.uk",
                ".gov.au",
                "who.int",
                "worldbank.org",
                "oecd.org",
            ]

            if any(
                domain in url
                for domain in government_domains
            ):
                score += 5

            government_keywords = [
                "government",
                "regulation",
                "regulations",
                "law",
                "legal",
                "legislation",
                "policy",
                "official",
                "public policy",
            ]

            keyword_matches = sum(
                1
                for keyword in government_keywords
                if keyword in text
            )

            score += min(
                keyword_matches,
                4,
            )

        # -----------------------------------------------------
        # RESEARCH
        # -----------------------------------------------------

        elif query_type == "research":

            research_domains = [
                ".edu",
                ".ac.uk",
                "nature.com",
                "sciencedirect.com",
                "springer.com",
                "pubmed.ncbi.nlm.nih.gov",
                "nih.gov",
                "arxiv.org",
            ]

            if any(
                domain in url
                for domain in research_domains
            ):
                score += 5

            research_keywords = [
                "research",
                "study",
                "journal",
                "scientific",
                "paper",
                "academic",
                "experiment",
                "survey",
            ]

            keyword_matches = sum(
                1
                for keyword in research_keywords
                if keyword in text
            )

            score += min(
                keyword_matches,
                4,
            )

        # -----------------------------------------------------
        # GENERAL
        # -----------------------------------------------------

        elif query_type == "general":

            general_keywords = [
                "official",
                "documentation",
                "guide",
                "reference",
                "information",
            ]

            keyword_matches = sum(
                1
                for keyword in general_keywords
                if keyword in text
            )

            score += min(
                keyword_matches,
                2,
            )

        return score

    # =========================================================
    # CACHE KEY
    # =========================================================

    @staticmethod
    def _cache_key(query: str) -> str:

        normalized = " ".join(
            query.lower().strip().split()
        )

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    # =========================================================
    # CACHE PATH
    # =========================================================

    def _cache_path(
        self,
        query: str,
    ) -> Path:

        return (
            self.cache_dir
            / f"{self._cache_key(query)}.json"
        )

    # =========================================================
    # LOAD CACHE
    # =========================================================

    def _load_cache(
        self,
        query: str,
    ):

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

                cached = json.load(
                    file
                )

        except (
            OSError,
            json.JSONDecodeError,
        ):
            return None

        timestamp = cached.get(
            "timestamp"
        )

        if timestamp is None:
            return None

        age = (
            time.time()
            - float(timestamp)
        )

        if age > self.cache_ttl:

            try:
                cache_path.unlink()

            except OSError:
                pass

            return None

        result = cached.get(
            "result"
        )

        if not isinstance(
            result,
            dict,
        ):
            return None

        result = dict(result)

        web_research = dict(
            result.get(
                "web_research",
                {},
            )
        )

        web_research[
            "cached"
        ] = True

        result[
            "web_research"
        ] = web_research

        return result

    # =========================================================
    # SAVE CACHE
    # =========================================================

    def _save_cache(
        self,
        query: str,
        result: dict,
    ):

        cache_path = self._cache_path(
            query
        )

        payload = {
            "timestamp": time.time(),
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

        except OSError:
            pass

    # =========================================================
    # RANK RESULTS
    # =========================================================

    @classmethod
    def _rank_results(
        cls,
        query: str,
        results: list[dict],
    ):

        query_type = (
            cls._detect_query_type(
                query
            )
        )

        print(
            f"🧭 Web Query Type: "
            f"{query_type}"
        )

        ranked_results = []

        for result in results:

            base_quality_score = (
                cls._source_quality_score(
                    result
                )
            )

            query_relevance_score = (
                cls._query_relevance_score(
                    query_type,
                    result,
                )
            )

            final_score = (
                base_quality_score
                + query_relevance_score
            )

            ranked_result = dict(
                result
            )

            # -------------------------------------------------
            # Score fields
            # -------------------------------------------------
            #
            # Keep both public and underscore versions.
            #
            # This makes the result backward-compatible
            # with existing tests and callers.

            ranked_result[
                "quality_score"
            ] = final_score

            ranked_result[
                "_quality_score"
            ] = final_score

            ranked_result[
                "_base_quality_score"
            ] = base_quality_score

            ranked_result[
                "base_quality_score"
            ] = base_quality_score

            ranked_result[
                "_query_relevance_score"
            ] = query_relevance_score

            ranked_result[
                "query_relevance_score"
            ] = query_relevance_score

            ranked_results.append(
                ranked_result
            )

        ranked_results.sort(
            key=lambda item: item.get(
                "quality_score",
                0,
            ),
            reverse=True,
        )

        return ranked_results

    # =========================================================
    # FETCH SOURCE
    # =========================================================

    def _fetch_source(
        self,
        result: dict,
    ) -> dict:

        url = result.get(
            "url",
            "",
        )

        title = result.get(
            "title",
            "",
        )

        snippet = result.get(
            "snippet",
            "",
        )

        evidence = {
            "title": title,
            "url": url,
            "content": snippet,
            "snippet": snippet,
            "fetched": False,
            "quality_score": result.get(
                "quality_score",
                0,
            ),
            "_quality_score": result.get(
                "_quality_score",
                result.get(
                    "quality_score",
                    0,
                ),
            ),
            "_base_quality_score": result.get(
                "_base_quality_score",
                0,
            ),
            "_query_relevance_score": result.get(
                "_query_relevance_score",
                0,
            ),
        }

        if not url:
            return evidence

        try:

            content = (
                self.fetcher.fetch(
                    url
                )
            )

            if content:

                evidence[
                    "content"
                ] = content

                evidence[
                    "fetched"
                ] = True

        except Exception as exc:

            print(
                f"⚠️ Could not fetch source "
                f"{url}: {exc}"
            )

        return evidence

    # =========================================================
    # COLLECT EVIDENCE
    # =========================================================

    def collect_evidence(
        self,
        query: str,
    ) -> dict:

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        query = query.strip()

        # -----------------------------------------------------
        # CACHE
        # -----------------------------------------------------

        cached = self._load_cache(
            query
        )

        if cached is not None:

            print(
                "⚡ Web research cache hit"
            )

            return cached

        # -----------------------------------------------------
        # SEARCH
        # -----------------------------------------------------

        try:

            search_results = (
                self.searcher.search(
                    query
                )
            )

        except Exception as exc:

            return {
                "query": query,
                "evidence": [],
                "sources": [],
                "web_research": {
                    "available": False,
                    "sources_found": 0,
                    "error": str(exc),
                    "cached": False,
                },
            }

        if not search_results:

            return {
                "query": query,
                "evidence": [],
                "sources": [],
                "web_research": {
                    "available": False,
                    "sources_found": 0,
                    "error": (
                        "No search results found."
                    ),
                    "cached": False,
                },
            }

        # -----------------------------------------------------
        # RANK
        # -----------------------------------------------------

        ranked_results = (
            self._rank_results(
                query,
                search_results,
            )
        )

        # -----------------------------------------------------
        # FETCH TOP RESULTS ONLY
        # -----------------------------------------------------

        fetch_count = min(
            self.fetch_top_k,
            len(ranked_results),
        )

        top_results = ranked_results[
            :fetch_count
        ]

        evidence = []

        for result in top_results:

            fetched_result = (
                self._fetch_source(
                    result
                )
            )

            evidence.append(
                fetched_result
            )

        # -----------------------------------------------------
        # SOURCE METADATA
        # -----------------------------------------------------

        sources = []

        for result in ranked_results:

            source_url = result.get(
                "url",
                "",
            )

            fetched = any(
                item.get("url")
                == source_url
                and item.get("fetched")
                for item in evidence
            )

            sources.append(
                {
                    "title": result.get(
                        "title",
                        "",
                    ),
                    "url": source_url,
                    "fetched": fetched,

                    "quality_score": result.get(
                        "quality_score",
                        0,
                    ),

                    "_quality_score": result.get(
                        "_quality_score",
                        result.get(
                            "quality_score",
                            0,
                        ),
                    ),

                    "_base_quality_score": result.get(
                        "_base_quality_score",
                        0,
                    ),

                    "base_quality_score": result.get(
                        "base_quality_score",
                        result.get(
                            "_base_quality_score",
                            0,
                        ),
                    ),

                    "_query_relevance_score": result.get(
                        "_query_relevance_score",
                        0,
                    ),

                    "query_relevance_score": result.get(
                        "query_relevance_score",
                        result.get(
                            "_query_relevance_score",
                            0,
                        ),
                    ),
                }
            )

        # -----------------------------------------------------
        # FINAL RESULT
        # -----------------------------------------------------

        result = {
            "query": query,
            "evidence": evidence,
            "sources": sources,
            "web_research": {
                "available": True,
                "sources_found": len(
                    sources
                ),
                "error": None,
                "cached": False,
            },
        }

        # -----------------------------------------------------
        # SAVE CACHE
        # -----------------------------------------------------

        self._save_cache(
            query,
            result,
        )

        return result

    # =========================================================
    # PUBLIC RESEARCH METHOD
    # =========================================================

    def research(
        self,
        query: str,
    ) -> dict:

        return self.collect_evidence(
            query
        )