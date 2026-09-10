from __future__ import annotations

import requests
from bs4 import BeautifulSoup


class WebFetcher:
    """
    Fetches web pages and extracts readable text.

    The fetcher is intentionally conservative:
    - Uses a normal browser-like User-Agent.
    - Applies a timeout.
    - Removes scripts/styles/navigation noise.
    - Limits extracted text to avoid enormous prompts.
    """

    def __init__(self, timeout=10, max_chars=6000):
        self.timeout = timeout
        self.max_chars = max_chars

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/139.0 Safari/537.36"
            )
        }

    def fetch(self, url: str) -> str:
        if not url or not url.strip():
            raise ValueError("url must not be empty")

        try:
            response = requests.get(
                url.strip(),
                headers=self.headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to fetch web page: {exc}"
            ) from exc

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove elements that usually do not contain useful article text.
        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "nav",
                "footer",
                "header",
                "form",
                "aside",
            ]
        ):
            element.decompose()

        text = soup.get_text(separator=" ", strip=True)

        # Normalize excessive whitespace.
        text = " ".join(text.split())

        if not text:
            raise RuntimeError("No readable text found on web page.")

        return text[: self.max_chars]