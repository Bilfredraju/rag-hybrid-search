from ddgs import DDGS


class WebSearch:
    """
    Performs live web searches.

    Returns structured search results containing
    title, URL, and snippet information.
    """

    def __init__(self, max_results=5):
        self.max_results = max_results

    def search(self, query):
        """
        Search the live web.

        Args:
            query: User search query.

        Returns:
            List of search result dictionaries.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        query = query.strip()

        try:
            with DDGS() as ddgs:

                results = list(
                    ddgs.text(
                        query,
                        max_results=self.max_results,
                        safesearch="moderate",
                    )
                )

        except Exception as exc:
            raise RuntimeError(
                f"Web search failed: {exc}"
            ) from exc

        normalized_results = []

        for result in results:

            normalized_results.append(
                {
                    "title": result.get(
                        "title",
                        "",
                    ),
                    "url": result.get(
                        "href",
                        "",
                    ),
                    "snippet": result.get(
                        "body",
                        "",
                    ),
                }
            )

        return normalized_results