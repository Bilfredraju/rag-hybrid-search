from src.assistant.assistant_pipeline import AssistantPipeline


def main():
    print("=" * 60)
    print("BOTH ROUTE TEST")
    print("=" * 60)

    assistant = AssistantPipeline()

    query = (
        "Based on Project Phoenix, how does it compare "
        "with the latest AI assistant trends?"
    )

    result = assistant.ask(query)

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nROUTE:")
    print(result["route"])

    print("\nCONFIDENCE:")
    print(result["confidence"])

    print("\nSOURCES:")

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(
            f"{index}. {source}"
        )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert result["question"] == query

    assert result["route"] == "both"

    assert result["answer"]

    assert result["sources"]

    # Must contain document evidence.
    assert any(
        source.get("type") == "document"
        for source in result["sources"]
    ), "No document source found."

    # Must contain web evidence.
    assert any(
        source.get("type") == "web"
        for source in result["sources"]
    ), "No web source found."

    # --------------------------------------------------
    # Check for fake citation markers.
    # --------------------------------------------------

    forbidden_markers = [
        "【1",
        "【2",
        "【3",
        "[1]",
        "[2]",
        "[3]",
        "†L1",
        "†L2",
        "†L3",
    ]

    for marker in forbidden_markers:
        assert marker not in result["answer"], (
            f"Fake citation marker detected: {marker}"
        )

    print("\n✅ BOTH ROUTE PASSED")


if __name__ == "__main__":
    main()