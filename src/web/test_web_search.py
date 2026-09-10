from src.web.web_research import WebResearch


def main():
    print("=" * 60)
    print("WEB RESEARCH TEST")
    print("=" * 60)

    researcher = WebResearch(max_results=5)

    query = "latest artificial intelligence news"

    result = researcher.research(query)

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(
            f"{index}. "
            f"{source['title']} "
            f"({source['url']}) "
            f"[fetched={source['fetched']}]"
        )

    assert result["question"] == query
    assert result["answer"]
    assert result["sources"]

    # Make sure the model did not create fake citation markers.
    forbidden_markers = [
        "【1",
        "【2",
        "【3",
        "[1]",
        "[2]",
        "[3]",
        "†L1",
        "†L2",
    ]

    for marker in forbidden_markers:
        assert marker not in result["answer"], (
            f"Fake citation marker detected: {marker}"
        )

    print("\n✅ WEB RESEARCH PASSED")


if __name__ == "__main__":
    main()