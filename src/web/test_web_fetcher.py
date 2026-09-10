from src.web.web_fetcher import WebFetcher


def main():
    print("=" * 60)
    print("WEB FETCHER TEST")
    print("=" * 60)

    fetcher = WebFetcher()

    url = "https://www.python.org/"

    text = fetcher.fetch(url)

    print("\nFetched characters:", len(text))
    print("\nPreview:")
    print(text[:500])

    assert text
    assert len(text) > 100

    print("\n✅ WEB FETCHER PASSED")


if __name__ == "__main__":
    main()