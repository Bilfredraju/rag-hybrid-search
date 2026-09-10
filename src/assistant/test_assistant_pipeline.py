from src.assistant.assistant_pipeline import (
    AssistantPipeline,
)


def test_general(assistant):

    result = assistant.ask(
        "What is the capital of France?"
    )

    print("\nGENERAL")
    print(result)

    assert result["route"] == "general"
    assert "paris" in result["answer"].lower()


def test_document(assistant):

    result = assistant.ask(
        "Who manages Project Phoenix?"
    )

    print("\nDOCUMENT")
    print(result)

    assert result["route"] == "document"

    assert any(
        source["source"]
        == "Project_Phoenix.pdf"
        for source in result["sources"]
    )


def test_current(assistant):

    result = assistant.ask(
        "What is the latest AI news?"
    )

    print("\nCURRENT")
    print(result)

    assert result["route"] == "current"

    assert result["answer"]

    assert result["sources"]

    for source in result["sources"]:
        assert source["url"]


def main():

    print("=" * 60)
    print("UNIVERSAL ASSISTANT TEST")
    print("=" * 60)

    assistant = AssistantPipeline()

    test_general(assistant)
    print("✅ GENERAL PASSED")

    test_document(assistant)
    print("✅ DOCUMENT PASSED")

    test_current(assistant)
    print("✅ CURRENT PASSED")

    print("\n" + "=" * 60)
    print("ALL ASSISTANT ROUTES PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()