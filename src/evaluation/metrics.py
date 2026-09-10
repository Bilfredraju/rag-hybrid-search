from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    total: int = 0
    routing_correct: int = 0
    source_correct: int = 0
    answer_success: int = 0
    successful_cases: int = 0

    total_latency: float = 0.0

    @property
    def routing_accuracy(self):
        if self.total == 0:
            return 0.0
        return self.routing_correct / self.total

    @property
    def source_accuracy(self):
        if self.total == 0:
            return 0.0
        return self.source_correct / self.total

    @property
    def answer_success_rate(self):
        if self.total == 0:
            return 0.0
        return self.answer_success / self.total

    @property
    def overall_success_rate(self):
        if self.total == 0:
            return 0.0
        return self.successful_cases / self.total

    @property
    def average_latency(self):
        if self.total == 0:
            return 0.0
        return self.total_latency / self.total

    def to_dict(self):
        return {
            "total": self.total,
            "routing_accuracy": self.routing_accuracy,
            "source_accuracy": self.source_accuracy,
            "answer_success_rate": self.answer_success_rate,
            "overall_success_rate": self.overall_success_rate,
            "average_latency_seconds": self.average_latency,
        }