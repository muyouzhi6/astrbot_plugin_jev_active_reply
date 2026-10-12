"""Hourly Jev call metrics without request content or credentials."""

from collections import Counter
from math import ceil


class Metrics:
    def __init__(self):
        self.decisions = self.replies = self.failures = 0
        self.latencies = Counter()
        self.errors = Counter()
        self.cancelled = 0
        self.outcomes = Counter()
        self.transport_streak = 0

    def decision(self, latency_ms, error=None):
        self.decisions += 1
        self.latencies[max(0, min(180000, round(latency_ms)))] += 1
        self.cancelled += error == "cancelled"
        self.failures += error is not None and error != "cancelled"
        if error is not None and error != "cancelled":
            self.errors[error] += 1
        if error == "transport_uncertain":
            self.transport_streak += 1
        elif error not in {"no_capacity", "queue_full"}:
            self.transport_streak = 0
        return error == "transport_uncertain" and self.transport_streak == 3

    def take(self):
        rank, total, p95 = ceil(self.decisions * 0.95), 0, 0
        for latency, count in sorted(self.latencies.items()):
            total += count
            if total >= rank:
                p95 = latency
                break
        result = self.decisions, self.replies, p95, self.failures
        self.decisions = self.replies = self.failures = 0
        self.latencies.clear()
        self.errors.clear()
        self.cancelled = 0
        self.outcomes.clear()
        return result
