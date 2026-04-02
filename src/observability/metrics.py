import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger("metrics")

class MetricsManager:
    """
    Simulated Prometheus Metrics Collector.
    Tracks internal counters and histograms for the RAG pipeline.
    """
    def __init__(self):
        # In-memory storage for counters or gauges before actual Prometheus push
        self._counters: Dict[str, int] = {
            "jobs_total": 0,
            "jobs_success": 0,
            "jobs_failed": 0,
            "chunks_processed": 0,
            "api_requests": 0
        }
        self._histograms: Dict[str, List[float]] = {
            "processing_latency_seconds": [],
            "retrieval_latency_seconds": []
        }
        logger.info("Metrics Manager started")

    def increment_counter(self, name: str, value: int = 1):
        if name in self._counters:
            self._counters[name] += value
            logger.debug(f"Counter {name} incremented by {value} (Total: {self._counters[name]})")

    def observe_hist(self, name: str, value: float):
        if name in self._histograms:
            self._histograms[name].append(value)
            # We only keep the last 100 observations in-memory in this dummy version
            if len(self._histograms[name]) > 100:
                self._histograms[name].pop(0)

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns a human-readable snapshot of the current application health.
        """
        summary = self._counters.copy()
        for hist, values in self._histograms.items():
            if values:
                summary[f"{hist}_avg"] = sum(values) / len(values)
                summary[f"{hist}_max"] = max(values)
            else:
                summary[f"{hist}_avg"] = 0.0
        return summary

metrics = MetricsManager()
