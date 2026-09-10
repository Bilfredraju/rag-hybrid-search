from __future__ import annotations

import time

from src.assistant.assistant_pipeline import AssistantPipeline
from src.evaluation.evaluation_dataset import EVALUATION_DATASET
from src.evaluation.metrics import EvaluationMetrics


class AssistantEvaluator:
    def __init__(self, assistant=None):
        self.assistant = assistant or AssistantPipeline()

    @staticmethod
    def _answer_success(result):
        answer = result.get("answer")

        if not answer:
            return False

        answer = str(answer).strip()

        if not answer:
            return False

        failure_phrases = [
            "i couldn't find",
            "i don't have enough information",
            "i could not find",
            "unable to answer",
        ]

        answer_lower = answer.lower()

        return not any(
            phrase in answer_lower
            for phrase in failure_phrases
        )

    @staticmethod
    def _source_success(result, expected_sources):
        if not expected_sources:
            return True

        sources = result.get("sources", [])

        if not sources:
            return False

        returned_sources = {
            str(source.get("source", "")).strip()
            for source in sources
            if source.get("source")
        }

        return all(
            expected_source in returned_sources
            for expected_source in expected_sources
        )

    def evaluate_case(self, case):
        start_time = time.perf_counter()

        try:
            result = self.assistant.ask(case.question)
            error = None
        except Exception as exc:
            result = {}
            error = str(exc)

        latency = time.perf_counter() - start_time

        actual_route = result.get("route")

        routing_correct = (
            actual_route == case.expected_route.value
        )

        source_correct = self._source_success(
            result,
            case.expected_sources,
        )

        answer_success = self._answer_success(result)

        case_success = (
            error is None
            and routing_correct
            and source_correct
            and answer_success
        )

        return {
            "question": case.question,
            "description": case.description,
            "expected_route": case.expected_route.value,
            "actual_route": actual_route,
            "routing_correct": routing_correct,
            "expected_sources": list(case.expected_sources),
            "source_correct": source_correct,
            "answer_success": answer_success,
            "case_success": case_success,
            "latency_seconds": latency,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "error": error,
        }

    def evaluate(self):
        metrics = EvaluationMetrics()
        results = []

        for case in EVALUATION_DATASET:
            print("\n" + "-" * 70)
            print(f"Question: {case.question}")
            print(f"Expected route: {case.expected_route.value}")

            result = self.evaluate_case(case)

            metrics.total += 1
            metrics.total_latency += result["latency_seconds"]

            if result["routing_correct"]:
                metrics.routing_correct += 1

            if result["source_correct"]:
                metrics.source_correct += 1

            if result["answer_success"]:
                metrics.answer_success += 1

            if result["case_success"]:
                metrics.successful_cases += 1

            results.append(result)

            print(
                f"Actual route : "
                f"{result['actual_route']}"
            )

            print(
                f"Routing      : "
                f"{'PASS' if result['routing_correct'] else 'FAIL'}"
            )

            print(
                f"Sources      : "
                f"{'PASS' if result['source_correct'] else 'FAIL'}"
            )

            print(
                f"Answer       : "
                f"{'PASS' if result['answer_success'] else 'FAIL'}"
            )

            print(
                f"Latency      : "
                f"{result['latency_seconds']:.2f}s"
            )

            if result["error"]:
                print(f"Error        : {result['error']}")

        return {
            "metrics": metrics,
            "results": results,
        }

    @staticmethod
    def print_report(evaluation):
        metrics = evaluation["metrics"]

        print("\n")
        print("=" * 70)
        print("UNIVERSAL AI ASSISTANT EVALUATION")
        print("=" * 70)

        print(f"\nTotal Questions       : {metrics.total}")

        print(
            f"Routing Accuracy      : "
            f"{metrics.routing_accuracy * 100:.2f}%"
        )

        print(
            f"Source Accuracy       : "
            f"{metrics.source_accuracy * 100:.2f}%"
        )

        print(
            f"Answer Success Rate   : "
            f"{metrics.answer_success_rate * 100:.2f}%"
        )

        print(
            f"Overall Success Rate  : "
            f"{metrics.overall_success_rate * 100:.2f}%"
        )

        print(
            f"Average Latency       : "
            f"{metrics.average_latency:.2f}s"
        )

        print("\n" + "=" * 70)
        print("CASE RESULTS")
        print("=" * 70)

        for index, result in enumerate(
            evaluation["results"],
            start=1,
        ):
            status = (
                "PASS"
                if result["case_success"]
                else "FAIL"
            )

            print(
                f"\n{index}. [{status}] "
                f"{result['question']}"
            )

            if result["error"]:
                print(
                    f"   Error: {result['error']}"
                )

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)