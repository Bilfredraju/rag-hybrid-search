from __future__ import annotations

import requests
from bs4 import BeautifulSoup

from src.config import WEB_FETCH_TIMEOUT, WEB_MAX_CHARS


class WebFetcher:
    """
    Fetches readable text from web pages.

    The timeout and maximum extracted content size
    are controlled through project configuration.
    """

    def __init__(
        self,
        timeout: int | None = None,
        max_chars: int | None = None,
    ):
        self.timeout = (
            timeout
            if timeout is not None
            else WEB_FETCH_TIMEOUT
        )

        self.max_chars = (
            max_chars
            if max_chars is not None
            else WEB_MAX_CHARS
        )

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
            raise ValueError(
                "url must not be empty"
            )

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

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        # Remove elements that generally do not contain
        # useful article/content text.
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

        text = soup.get_text(
            separator=" ",
            strip=True,
        )

        text = " ".join(
            text.split()
        )

        if not text:
            raise RuntimeError(
                "No readable text found on web page."
            )

        return text[: self.max_chars]