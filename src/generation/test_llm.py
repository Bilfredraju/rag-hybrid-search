from src.generation.llm import LLM


def main():

    llm = LLM()

    answer = llm.generate(
        "Who are you? Reply in one sentence."
    )

    print(answer)


if __name__ == "__main__":
    main()