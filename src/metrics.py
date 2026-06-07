"""Lightweight in-process metrics for RetServe."""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict

LATENCY_BUCKETS_SECONDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def _labels_to_text(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    body = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    return f"{{{body}}}"


@dataclass
class Histogram:
    """Minimal cumulative Prometheus-style histogram."""

    buckets: tuple[float, ...] = LATENCY_BUCKETS_SECONDS
    counts: list[int] = field(init=False)
    total: float = 0.0
    observations: int = 0

    def __post_init__(self) -> None:
        self.counts = [0 for _ in self.buckets]

    def observe(self, value: float) -> None:
        """Record one observed value in seconds."""
        self.observations += 1
        self.total += value
        for index, bucket in enumerate(self.buckets):
            if value <= bucket:
                self.counts[index] += 1

    def render(self, name: str, labels: dict[str, str] | None = None) -> list[str]:
        """Render histogram lines in Prometheus text format."""
        labels = labels or {}
        lines: list[str] = []
        for bucket, count in zip(self.buckets, self.counts):
            bucket_labels = {**labels, "le": f"{bucket:g}"}
            lines.append(f"{name}_bucket{_labels_to_text(bucket_labels)} {count}")
        lines.append(
            f"{name}_bucket{_labels_to_text({**labels, 'le': '+Inf'})} "
            f"{self.observations}"
        )
        lines.append(f"{name}_sum{_labels_to_text(labels)} {self.total:.9f}")
        lines.append(f"{name}_count{_labels_to_text(labels)} {self.observations}")
        return lines


class MetricsRegistry:
    """Small thread-safe registry for service counters, gauges, and histograms."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        self._counters: DefaultDict[tuple[str, tuple[tuple[str, str], ...]], int] = (
            defaultdict(int)
        )
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            Histogram,
        ] = {}

    @property
    def enabled(self) -> bool:
        """Return whether metrics collection is enabled."""
        return self._enabled

    def increment(
        self,
        name: str,
        amount: int = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter."""
        if not self._enabled:
            return
        key = self._metric_key(name, labels)
        with self._lock:
            self._counters[key] += amount

    def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Set a gauge value."""
        if not self._enabled:
            return
        key = self._metric_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram observation in seconds."""
        if not self._enabled:
            return
        key = self._metric_key(name, labels)
        with self._lock:
            histogram = self._histograms.setdefault(key, Histogram())
            histogram.observe(value)

    def render_prometheus(self) -> str:
        """Render collected metrics using Prometheus text exposition format."""
        if not self._enabled:
            return "# RetServe metrics are disabled\n"

        lines = [
            "# HELP retserve_requests_total Total retrieval HTTP requests.",
            "# TYPE retserve_requests_total counter",
        ]

        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = dict(self._histograms)

        for (name, labels), value in sorted(counters.items()):
            lines.append(f"{name}{_labels_to_text(dict(labels))} {value}")

        lines.extend(
            [
                "# HELP retserve_ready Service readiness state.",
                "# TYPE retserve_ready gauge",
            ]
        )
        for (name, labels), value in sorted(gauges.items()):
            lines.append(f"{name}{_labels_to_text(dict(labels))} {value:g}")

        for (name, labels), histogram in sorted(histograms.items()):
            lines.append(f"# TYPE {name} histogram")
            lines.extend(histogram.render(name, dict(labels)))

        return "\n".join(lines) + "\n"

    @staticmethod
    def _metric_key(
        name: str,
        labels: dict[str, str] | None = None,
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        label_items = tuple(sorted((labels or {}).items()))
        return name, label_items
