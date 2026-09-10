from src.assistant.assistant_pipeline import AssistantPipeline


def main():
    print("=" * 60)
    print("WEB QUERY BUILDER TEST")
    print("=" * 60)

    query = (
        "Based on Project Phoenix, how does it compare "
        "with the latest AI assistant trends?"
    )

    web_query = (
        AssistantPipeline._build_web_query(
            query
        )
    )

    print("\nOriginal query:")
    print(query)

    print("\nGenerated web query:")
    print(web_query)

    assert web_query

    # The internal project name should not dominate
    # the external web search.
    assert "project phoenix" not in web_query.lower()

    # The current-trend intent should remain.
    assert "latest" in web_query.lower()

    assert "ai assistant" in web_query.lower()

    print("\n✅ WEB QUERY BUILDER PASSED")


if __name__ == "__main__":
    main()