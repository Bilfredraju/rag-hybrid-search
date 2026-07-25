from src.pipeline.rag_pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline()

    question = "What is the leave policy?"

    result = pipeline.ask(question)

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(result["question"])

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result["sources"]:
        print(source)


if __name__ == "__main__":
    main()