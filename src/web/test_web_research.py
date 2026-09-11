from src.web.web_research import WebResearch


def main():
    print("=" * 60)
    print("QUERY-AWARE WEB RESEARCH TEST")
    print("=" * 60)

    researcher = WebResearch()

    test_queries = [
        "latest artificial intelligence news",
        "current Bitcoin price",
        "latest employee leave management trends",
        "latest government regulations",
        "latest scientific research on machine learning",
    ]

    print("\nQUERY TYPES")
    print("-" * 60)

    for query in test_queries:
        query_type = researcher._detect_query_type(
            query
        )

        print(
            f"{query}\n"
            f"  → {query_type}"
        )

    print("\nSOURCE SCORING")
    print("-" * 60)

    test_results = [
        {
            "title": "Reuters AI News",
            "url": (
                "https://www.reuters.com/"
                "technology/artificial-intelligence/"
            ),
            "snippet": "Latest AI news",
        },
        {
            "title": "TechCrunch AI",
            "url": (
                "https://techcrunch.com/"
                "category/artificial-intelligence/"
            ),
            "snippet": "AI technology news",
        },
        {
            "title": "Random AI Blog",
            "url": (
                "https://example.com/"
                "ai-news"
            ),
            "snippet": "AI news",
        },
    ]

    ranked = researcher._rank_results(
        "latest artificial intelligence news",
        test_results,
    )

    for index, result in enumerate(
        ranked,
        start=1,
    ):
        print(
            f"\n{index}. "
            f"{result['title']}"
        )

        print(
            f"URL: {result['url']}"
        )

        print(
            f"Base Quality: "
            f"{result['_base_quality_score']}"
        )

        print(
            f"Query Relevance: "
            f"{result['_query_relevance_score']}"
        )

        print(
            f"Final Score: "
            f"{result['_quality_score']}"
        )

    print("\n" + "=" * 60)
    print("QUERY-AWARE WEB RESEARCH TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()