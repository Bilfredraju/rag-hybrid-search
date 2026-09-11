from src.web.web_research import WebResearch


def main():
    print("=" * 60)
    print("WEB RESEARCH TEST")
    print("=" * 60)

    query = "latest artificial intelligence news"

    researcher = WebResearch(
        max_results=5
    )

    result = researcher.research(
        query
    )

    print("\nQUESTION:")
    print(query)

    print("\nWEB RESEARCH STATUS:")
    print(
        result["web_research"]
    )

    print("\nSOURCES FOUND:")
    print(
        len(result["sources"])
    )

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(
            f"\n{index}. {source['title']}"
        )
        print(
            f"URL: {source['url']}"
        )
        print(
            f"Fetched: {source['fetched']}"
        )
        print(
            f"Quality Score: "
            f"{source.get('quality_score', 0)}"
        )

    print("\nEVIDENCE:")
    print("-" * 60)

    for index, evidence in enumerate(
        result["evidence"],
        start=1,
    ):
        print(
            f"\n[{index}] "
            f"{evidence['title']}"
        )

        print(
            f"URL: {evidence['url']}"
        )

        print(
            f"Fetched: "
            f"{evidence['fetched']}"
        )

        print(
            f"Quality Score: "
            f"{evidence.get('quality_score', 0)}"
        )

        content = evidence.get(
            "content",
            "",
        )

        print(
            f"Content: "
            f"{content[:500]}..."
        )

    print("\n" + "=" * 60)
    print("WEB RESEARCH TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()