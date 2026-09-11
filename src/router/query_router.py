from enum import Enum
import re


class QueryRoute(str, Enum):
    """
    Supported query execution routes.
    """

    DOCUMENT = "document"
    GENERAL = "general"
    CURRENT = "current"
    BOTH = "both"


class QueryRouter:
    """
    Classifies user queries into execution routes.

    DOCUMENT:
        The answer should primarily come from
        indexed company/user documents.

    GENERAL:
        The question can be answered using
        general LLM knowledge.

    CURRENT:
        The question requires up-to-date information.

    BOTH:
        The question requires indexed documents
        plus general/current information.
    """

    CURRENT_PATTERNS = [
        r"\bcurrent\b",
        r"\bcurrently\b",
        r"\bnow\b",
        r"\btoday\b",
        r"\blatest\b",
        r"\brecent\b",
        r"\brecently\b",
        r"\bthis week\b",
        r"\bthis month\b",
        r"\bthis year\b",
        r"\bas of\b",
        r"\bup[- ]to[- ]date\b",
        r"\breal[- ]time\b",
        r"\bright now\b",
        r"\bnews\b",
        r"\bprice\b",
        r"\bstock price\b",
        r"\bweather\b",
    ]

    DOCUMENT_PATTERNS = [
        r"\baccording to (the|my|our)\b",
        r"\bin (the|this|my|our)\b.*\b(document|pdf|file|policy|handbook|report)\b",
        r"\bfrom (the|this|my|our)\b.*\b(document|pdf|file|policy|handbook|report)\b",
        r"\bwhat does (the|this|my|our)\b",
        r"\bwhat is (the|this|my|our)\b.*\b(policy|rule|procedure)\b",
        r"\bwho manages\b",
        r"\bwho is responsible for\b",
        r"\bproject phoenix\b",
        r"\borion cnc\b",
        r"\bleave policy\b",
        r"\bcompany policy\b",
        r"\bemployee policy\b",
    ]

    BOTH_PATTERNS = [
        r"\bbased on\b",
        r"\baccording to\b.*\bcurrent\b",
        r"\baccording to\b.*\blatest\b",
        r"\busing\b.*\bcurrent\b",
        r"\bcompare\b.*\bwith\b",
        r"\bhow does\b.*\bcompare\b",
        r"\bin .* documents\b.*\btoday\b",
        r"\bin .* documents\b.*\bcurrently\b",
        r"\bfrom .* policy\b.*\bcurrent\b",
    ]

    def classify(self, query: str) -> QueryRoute:
        """
        Classify a user query.

        The routing order is:

            BOTH
              ↓
            CURRENT
              ↓
            DOCUMENT
              ↓
            GENERAL
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        normalized = " ".join(
            query.lower().strip().split()
        )

        if self._matches(
            normalized,
            self.BOTH_PATTERNS,
        ):
            return QueryRoute.BOTH

        if self._matches(
            normalized,
            self.CURRENT_PATTERNS,
        ):
            return QueryRoute.CURRENT

        if self._matches(
            normalized,
            self.DOCUMENT_PATTERNS,
        ):
            return QueryRoute.DOCUMENT

        return QueryRoute.GENERAL

    @staticmethod
    def _matches(
        query: str,
        patterns: list[str],
    ) -> bool:
        """
        Return True when any routing pattern
        matches the query.
        """

        return any(
            re.search(
                pattern,
                query,
            )
            for pattern in patterns
        )