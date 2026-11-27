"""
Unit tests for benchmark utilities and results/benchmark.json schema.
"""

import os
import json
from src.benchmark import get_machine_specs


def test_get_machine_specs():
    specs = get_machine_specs()
    assert "os" in specs
    assert "cpu_model" in specs
    assert "cpu_cores_logical" in specs
    assert specs["cpu_cores_logical"] >= 1
    assert "kafka_broker" in specs


def test_benchmark_json_schema():
    path = "results/benchmark.json"
    assert os.path.exists(path), "results/benchmark.json does not exist"

    with open(path) as f:
        data = json.load(f)

    assert "machine_specs" in data
    assert "results" in data
    assert len(data["results"]) >= 4

    for entry in data["results"]:
        assert "target_rate_msgs_per_sec" in entry
        assert "producer_achieved_rate" in entry
        assert "consumer_achieved_throughput" in entry
        assert "latency_ms" in entry
        for pct in ("p50", "p95", "p99"):
            assert pct in entry["latency_ms"]
        assert "consumer_lag_at_end" in entry
