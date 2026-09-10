from src.evaluation.evaluator import AssistantEvaluator


def main():
    evaluator = AssistantEvaluator()

    evaluation = evaluator.evaluate()

    evaluator.print_report(evaluation)


if __name__ == "__main__":
    main()