from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from pathlib import Path
from urllib.parse import urlparse

from src.config import (
    WEB_CACHE_DIR,
    WEB_CACHE_TTL,
    WEB_CACHE_TTL_FINANCE,
    WEB_CACHE_TTL_GENERAL,
    WEB_CACHE_TTL_GOVERNMENT,
    WEB_CACHE_TTL_HR,
    WEB_CACHE_TTL_NEWS,
    WEB_CACHE_TTL_RESEARCH,
    WEB_CACHE_TTL_TECHNOLOGY,
    WEB_DOMAIN_FAILURE_TTL,
    WEB_FETCH_TOP_K,
    WEB_FETCH_TIMEOUT,
    WEB_MAX_CHARS,
    WEB_MAX_DOMAIN_FAILURES,
    WEB_MAX_RESULTS,
    WEB_MAX_WORKERS,
    WEB_MIN_CONTENT_CHARS,
    WEB_RESEARCH_TIMEOUT,
)

from src.web.web_fetcher import WebFetcher
from src.web.web_search import WebSearch


class WebResearch:
    """
    Query-aware web research pipeline.

    Responsibilities:
    1. Search the web.
    2. Classify the query.
    3. Rank sources by quality + query relevance.
    4. Filter unreliable domains.
    5. Fetch reliable sources concurrently.
    6. Validate fetched content.
    7. Prefer verified page evidence.
    8. Gracefully fall back to search evidence when
       page fetching is unavailable.
    9. Track domain reliability.
    10. Cache successful research results.
    11. Enforce a bounded research timeout.

    Evidence levels:

    VERIFIED:
        Actual web page content was successfully fetched
        and validated.

    SEARCH_FALLBACK:
        Search engine title/snippet only.
        It is not treated as verified page content.
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
        research_timeout: int | None = None,
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

        self.research_timeout = (
            research_timeout
            if research_timeout is not None
            else WEB_RESEARCH_TIMEOUT
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

        # -----------------------------------------------------
        # DOMAIN RELIABILITY
        # -----------------------------------------------------

        self.domain_reliability_file = (
            self.cache_dir
            / "domain_reliability.json"
        )

        self.domain_reliability_lock = (
            threading.Lock()
        )

        self.domain_reliability = (
            self._load_domain_reliability()
        )

    # =========================================================
    # QUERY TYPE DETECTION
    # =========================================================

    @staticmethod
    def _detect_query_type(
        query: str,
    ) -> str:
        """
        Detect broad web-query category.

        Categories:
        - finance
        - news
        - government
        - hr
        - research
        - technology
        - general
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
            "retrieval-augmented generation",
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
    def _source_quality_score(
        result: dict,
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

        if url.startswith(
            "https://"
        ):
            score += 1

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
    # QUERY RELEVANCE
    # =========================================================

    @classmethod
    def _query_relevance_score(
        cls,
        query_type: str,
        result: dict,
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

            score += min(
                sum(
                    1
                    for keyword in finance_keywords
                    if keyword in text
                ),
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

            score += min(
                sum(
                    1
                    for keyword in news_keywords
                    if keyword in text
                ),
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

            score += min(
                sum(
                    1
                    for keyword in technology_keywords
                    if keyword in text
                ),
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

            score += min(
                sum(
                    1
                    for keyword in hr_keywords
                    if keyword in text
                ),
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

            score += min(
                sum(
                    1
                    for keyword in government_keywords
                    if keyword in text
                ),
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

            score += min(
                sum(
                    1
                    for keyword in research_keywords
                    if keyword in text
                ),
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

            score += min(
                sum(
                    1
                    for keyword in general_keywords
                    if keyword in text
                ),
                2,
            )

        return score

    # =========================================================
    # CACHE TTL
    # =========================================================

    @staticmethod
    def _cache_ttl_for_query_type(
        query_type: str,
    ) -> int:

        ttl_map = {
            "finance": WEB_CACHE_TTL_FINANCE,
            "news": WEB_CACHE_TTL_NEWS,
            "government": WEB_CACHE_TTL_GOVERNMENT,
            "hr": WEB_CACHE_TTL_HR,
            "technology": WEB_CACHE_TTL_TECHNOLOGY,
            "research": WEB_CACHE_TTL_RESEARCH,
            "general": WEB_CACHE_TTL_GENERAL,
        }

        return ttl_map.get(
            query_type,
            WEB_CACHE_TTL,
        )

    # =========================================================
    # CACHE
    # =========================================================

    @staticmethod
    def _cache_key(
        query: str,
    ) -> str:

        normalized = " ".join(
            query.lower().strip().split()
        )

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    def _cache_path(
        self,
        query: str,
    ) -> Path:

        return (
            self.cache_dir
            / f"{self._cache_key(query)}.json"
        )

    def _load_cache(
        self,
        query: str,
        cache_ttl: int | None = None,
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
                cached = json.load(file)

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

        try:
            age = (
                time.time()
                - float(timestamp)
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

        ttl = (
            self.cache_ttl
            if cache_ttl is None
            else cache_ttl
        )

        if ttl <= 0:
            return None

        if age > ttl:

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

    def _save_cache(
        self,
        query: str,
        result: dict,
    ) -> None:

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
    # DOMAIN RELIABILITY
    # =========================================================

    @staticmethod
    def _domain_from_url(
        url: str,
    ) -> str:

        if not url:
            return ""

        try:
            hostname = urlparse(
                url.strip()
            ).hostname

            if not hostname:
                return ""

            hostname = hostname.lower()

            if hostname.startswith(
                "www."
            ):
                hostname = hostname[4:]

            return hostname

        except Exception:
            return ""

    def _load_domain_reliability(
        self,
    ) -> dict:

        if not self.domain_reliability_file.exists():
            return {}

        try:
            with self.domain_reliability_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if isinstance(
                data,
                dict,
            ):
                return data

        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            pass

        return {}

    def _save_domain_reliability(
        self,
    ) -> None:

        try:
            self.domain_reliability_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_file = (
                self.domain_reliability_file.with_suffix(
                    ".tmp"
                )
            )

            with temporary_file.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    self.domain_reliability,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            temporary_file.replace(
                self.domain_reliability_file
            )

        except OSError:
            pass

    def _is_domain_unreliable(
        self,
        url: str,
    ) -> bool:

        domain = self._domain_from_url(
            url
        )

        if not domain:
            return False

        needs_save = False

        with self.domain_reliability_lock:

            record = (
                self.domain_reliability.get(
                    domain
                )
            )

            if not record:
                return False

            try:
                failures = int(
                    record.get(
                        "failures",
                        0,
                    )
                )

                blocked_until = float(
                    record.get(
                        "blocked_until",
                        0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                return False

            now = time.time()

            if (
                failures
                >= WEB_MAX_DOMAIN_FAILURES
                and blocked_until > now
            ):
                return True

            if (
                blocked_until
                and blocked_until <= now
            ):
                self.domain_reliability.pop(
                    domain,
                    None,
                )
                needs_save = True

        if needs_save:
            self._save_domain_reliability()

        return False

    def _record_domain_failure(
        self,
        url: str,
        error: str = "",
    ) -> None:

        domain = self._domain_from_url(
            url
        )

        if not domain:
            return

        reached_threshold = False

        with self.domain_reliability_lock:

            record = (
                self.domain_reliability.get(
                    domain,
                    {},
                )
            )

            try:
                failures = int(
                    record.get(
                        "failures",
                        0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                failures = 0

            failures += 1

            now = time.time()

            blocked_until = 0

            if (
                failures
                >= WEB_MAX_DOMAIN_FAILURES
            ):
                blocked_until = (
                    now
                    + WEB_DOMAIN_FAILURE_TTL
                )
                reached_threshold = True

            self.domain_reliability[
                domain
            ] = {
                "failures": failures,
                "last_failure": now,
                "blocked_until": blocked_until,
                "last_error": str(error)[:500],
            }

        self._save_domain_reliability()

        if reached_threshold:
            print(
                f"⛔ Temporarily avoiding "
                f"unreliable domain: "
                f"{domain} "
                f"({failures} failures)"
            )

    def _record_domain_success(
        self,
        url: str,
    ) -> None:

        domain = self._domain_from_url(
            url
        )

        if not domain:
            return

        removed = False

        with self.domain_reliability_lock:

            if (
                domain
                in self.domain_reliability
            ):
                self.domain_reliability.pop(
                    domain,
                    None,
                )
                removed = True

        if removed:
            self._save_domain_reliability()

    def _filter_unreliable_results(
        self,
        results: list[dict],
    ) -> list[dict]:

        reliable_results = []

        skipped = 0

        for result in results:

            url = result.get(
                "url",
                "",
            )

            if self._is_domain_unreliable(
                url
            ):
                skipped += 1
                continue

            reliable_results.append(
                result
            )

        if skipped:
            print(
                f"🛡️ Skipped {skipped} "
                f"temporarily unreliable "
                f"web source(s)"
            )

        return reliable_results

    # =========================================================
    # RANK RESULTS
    # =========================================================

    @classmethod
    def _rank_results(
        cls,
        query: str,
        results: list[dict],
    ) -> list[dict]:

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
    # FETCH VERIFIED SOURCE
    # =========================================================

    def _fetch_source(
        self,
        result: dict,
    ) -> dict | None:
        """
        Fetch one web page.

        A successful page fetch produces VERIFIED evidence.

        A failed page fetch returns None. The caller can then
        create a SEARCH_FALLBACK evidence item from the original
        search result.
        """

        url = result.get(
            "url",
            "",
        ).strip()

        title = result.get(
            "title",
            "",
        ).strip()

        if not url:
            return None

        if self._is_domain_unreliable(
            url
        ):
            return None

        try:

            content = self.fetcher.fetch(
                url
            )

            if (
                not content
                or not content.strip()
            ):
                raise RuntimeError(
                    "No readable text found on web page."
                )

            content = content.strip()

            if (
                len(content)
                < WEB_MIN_CONTENT_CHARS
            ):
                raise RuntimeError(
                    "Fetched web content is below "
                    "the minimum quality threshold "
                    f"({WEB_MIN_CONTENT_CHARS} chars)."
                )

            self._record_domain_success(
                url
            )

            return {
                "title": title,
                "url": url,
                "content": content,

                "snippet": result.get(
                    "snippet",
                    "",
                ),

                "fetched": True,
                "verified": True,
                "evidence_type": "verified",

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

        except Exception as exc:

            self._record_domain_failure(
                url,
                str(exc),
            )

            print(
                f"⚠️ Could not fetch source "
                f"{url}: {exc}"
            )

            return None

    # =========================================================
    # SEARCH FALLBACK
    # =========================================================

    @staticmethod
    def _build_search_fallback(
        result: dict,
    ) -> dict | None:
        """
        Build explicitly unverified evidence from the
        search-engine result.

        This is intentionally separate from verified page
        evidence.

        Search snippets are useful for graceful degradation,
        but the LLM must know that they were not verified
        by fetching the actual page.
        """

        title = (
            result.get(
                "title",
                "",
            ).strip()
        )

        url = (
            result.get(
                "url",
                "",
            ).strip()
        )

        snippet = (
            result.get(
                "snippet",
                "",
            ).strip()
        )

        if not title and not snippet:
            return None

        if not snippet:
            return None

        return {
            "title": title,
            "url": url,

            # Deliberately use the snippet as content only
            # inside an explicitly labeled fallback object.
            "content": snippet,

            "snippet": snippet,

            "fetched": False,
            "verified": False,
            "evidence_type": "search_fallback",

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

    # =========================================================
    # FETCH SOURCES CONCURRENTLY
    # =========================================================

    def _fetch_sources_with_timeout(
        self,
        candidates: list[dict],
        timeout: float,
    ) -> list[dict]:
        """
        Fetch candidates concurrently with a bounded wait.

        Futures that do not finish before the timeout are
        cancelled where possible.

        The main request never waits indefinitely for a slow
        website.
        """

        if not candidates:
            return []

        worker_count = max(
            1,
            min(
                self.max_workers,
                len(candidates),
            ),
        )

        executor = ThreadPoolExecutor(
            max_workers=worker_count
        )

        futures = [
            executor.submit(
                self._fetch_source,
                candidate,
            )
            for candidate in candidates
        ]

        verified_results = []

        deadline = (
            time.perf_counter()
            + max(
                0.1,
                float(timeout),
            )
        )

        try:

            for future in futures:

                remaining = (
                    deadline
                    - time.perf_counter()
                )

                if remaining <= 0:
                    break

                try:

                    result = future.result(
                        timeout=remaining
                    )

                    if result is not None:
                        verified_results.append(
                            result
                        )

                except FuturesTimeoutError:

                    print(
                        "⏱️ Web source fetch "
                        "exceeded research timeout."
                    )

                except Exception as exc:

                    print(
                        f"⚠️ Web fetch worker failed: "
                        f"{exc}"
                    )

        finally:

            # Do not wait for slow background requests.
            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

        return verified_results

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

        started_at = time.perf_counter()

        query_type = (
            self._detect_query_type(
                query
            )
        )

        cache_ttl = (
            self._cache_ttl_for_query_type(
                query_type
            )
        )

        # -----------------------------------------------------
        # CACHE
        # -----------------------------------------------------

        cached = self._load_cache(
            query,
            cache_ttl=cache_ttl,
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

            print(
                f"⚠️ Web search failed: "
                f"{exc}"
            )

            return {
                "query": query,
                "evidence": [],
                "sources": [],
                "web_research": {
                    "available": False,
                    "query_type": query_type,
                    "sources_found": 0,
                    "sources_fetched": 0,
                    "verified_sources": 0,
                    "search_fallback_sources": 0,
                    "error": str(exc),
                    "cached": False,
                    "latency": round(
                        time.perf_counter()
                        - started_at,
                        2,
                    ),
                },
            }

        if not search_results:

            return {
                "query": query,
                "evidence": [],
                "sources": [],
                "web_research": {
                    "available": False,
                    "query_type": query_type,
                    "sources_found": 0,
                    "sources_fetched": 0,
                    "verified_sources": 0,
                    "search_fallback_sources": 0,
                    "error": (
                        "No search results found."
                    ),
                    "cached": False,
                    "latency": round(
                        time.perf_counter()
                        - started_at,
                        2,
                    ),
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
        # RELIABILITY FILTER
        # -----------------------------------------------------

        reliable_results = (
            self._filter_unreliable_results(
                ranked_results
            )
        )

        # -----------------------------------------------------
        # FETCH CANDIDATES
        # -----------------------------------------------------

        fetch_count = min(
            self.fetch_top_k,
            len(reliable_results),
        )

        top_results = (
            reliable_results[
                :fetch_count
            ]
        )

        # -----------------------------------------------------
        # VERIFIED FETCH
        # -----------------------------------------------------

        verified_evidence = (
            self._fetch_sources_with_timeout(
                top_results,
                timeout=self.research_timeout,
            )
        )

        # -----------------------------------------------------
        # SEARCH FALLBACK
        # -----------------------------------------------------
        #
        # Use search snippets only for sources whose pages
        # could not be verified.
        #
        # Prefer verified evidence.
        # -----------------------------------------------------

        verified_urls = {
            item.get("url")
            for item in verified_evidence
            if item.get("url")
        }

        fallback_evidence = []

        for result in top_results:

            url = result.get(
                "url",
                "",
            )

            if url in verified_urls:
                continue

            fallback = (
                self._build_search_fallback(
                    result
                )
            )

            if fallback is not None:
                fallback_evidence.append(
                    fallback
                )

        # -----------------------------------------------------
        # FINAL EVIDENCE
        # -----------------------------------------------------
        #
        # Verified evidence always comes first.
        # Search fallback is explicitly unverified.
        # -----------------------------------------------------

        evidence = (
            verified_evidence
            + fallback_evidence
        )

        # -----------------------------------------------------
        # SOURCE METADATA
        # -----------------------------------------------------

        fetched_urls = {
            item.get("url")
            for item in verified_evidence
            if item.get("url")
        }

        fallback_urls = {
            item.get("url")
            for item in fallback_evidence
            if item.get("url")
        }

        sources = []

        for ranked_result in ranked_results:

            source_url = ranked_result.get(
                "url",
                "",
            )

            fetched = (
                source_url
                in fetched_urls
            )

            fallback = (
                source_url
                in fallback_urls
            )

            sources.append(
                {
                    "title": ranked_result.get(
                        "title",
                        "",
                    ),

                    "url": source_url,

                    "fetched": fetched,

                    "verified": fetched,

                    "evidence_type": (
                        "verified"
                        if fetched
                        else (
                            "search_fallback"
                            if fallback
                            else "unavailable"
                        )
                    ),

                    "quality_score": ranked_result.get(
                        "quality_score",
                        0,
                    ),

                    "_quality_score": ranked_result.get(
                        "_quality_score",
                        ranked_result.get(
                            "quality_score",
                            0,
                        ),
                    ),

                    "_base_quality_score": ranked_result.get(
                        "_base_quality_score",
                        0,
                    ),

                    "base_quality_score": ranked_result.get(
                        "base_quality_score",
                        ranked_result.get(
                            "_base_quality_score",
                            0,
                        ),
                    ),

                    "_query_relevance_score": ranked_result.get(
                        "_query_relevance_score",
                        0,
                    ),

                    "query_relevance_score": ranked_result.get(
                        "query_relevance_score",
                        ranked_result.get(
                            "_query_relevance_score",
                            0,
                        ),
                    ),
                }
            )

        verified_count = len(
            verified_evidence
        )

        fallback_count = len(
            fallback_evidence
        )

        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------

        result = {
            "query": query,

            "evidence": evidence,

            "sources": sources,

            "web_research": {
                "available": bool(
                    evidence
                ),

                "query_type": query_type,

                "sources_found": len(
                    ranked_results
                ),

                "sources_fetched": verified_count,

                "verified_sources": verified_count,

                "search_fallback_sources": (
                    fallback_count
                ),

                "error": (
                    None
                    if evidence
                    else (
                        "No web evidence "
                        "was available."
                    )
                ),

                "cached": False,

                "latency": round(
                    time.perf_counter()
                    - started_at,
                    2,
                ),
            },
        }

        # -----------------------------------------------------
        # CACHE
        # -----------------------------------------------------
        #
        # Cache only results that contain evidence.
        # This prevents transient web failures from poisoning
        # the cache.
        # -----------------------------------------------------

        if evidence:
            self._save_cache(
                query,
                result,
            )

        return result

    # =========================================================
    # PUBLIC API
    # =========================================================

    def research(
        self,
        query: str,
    ) -> dict:

        return self.collect_evidence(
            query
        )