"""
Streaming throughput and end-to-end latency benchmark (benchmark.py).
Replays flows concurrently with the scoring consumer across multiple target rates
and records achieved throughput, p50/p95/p99 latency, consumer lag, and machine specs
to results/benchmark.json.

Reproducible command:
    python src/benchmark.py
"""

import os
import sys
import json
import time
import uuid
import platform
import subprocess
import threading
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from confluent_kafka.admin import AdminClient, NewTopic

from src.producer import replay_flows
from src.consumer import consume_and_score

DEFAULT_BOOTSTRAP = "localhost:9092"
DEFAULT_RATES = [100, 500, 1000, 2000, 5000]
DEFAULT_OUTPUT = "results/benchmark.json"


def get_machine_specs() -> dict:
    """Collect host CPU, RAM, OS, and Kafka broker metadata."""
    cpu_model = platform.processor() or "unknown"
    ram_gb = None

    if platform.system() == "Darwin":
        try:
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
        try:
            mem_bytes = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            )
            ram_gb = round(mem_bytes / (1024 ** 3), 2)
        except Exception:
            pass

    return {
        "os": platform.platform(),
        "arch": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_cores_logical": os.cpu_count(),
        "ram_gb": ram_gb,
        "python_version": platform.python_version(),
        "kafka_broker": "single-broker apache/kafka:3.7.0 (KRaft) via Docker Compose",
    }


def ensure_topics(bootstrap_servers: str, topics: list[str]) -> None:
    """Pre-create Kafka topics so partition creation overhead does not skew first-message latency."""
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    new_topics = [NewTopic(t, num_partitions=1, replication_factor=1) for t in topics]
    futures = admin.create_topics(new_topics)
    for topic, fut in futures.items():
        try:
            fut.result(timeout=10)
        except Exception:
            # Topic may already exist
            pass
    # Brief pause to allow metadata propagation
    time.sleep(0.5)


def run_single_rate_benchmark(
    rate: float,
    duration_sec: float = 8.0,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    model_name: str = "xgboost_baseline",
    batch_size: int = 64,
    run_id: str | None = None,
) -> dict:
    """Run a concurrent producer + consumer benchmark at a single target rate."""
    run_id = run_id or uuid.uuid4().hex[:8]
    flows_topic = f"bench-flows-{run_id}-{int(rate)}"
    alerts_topic = f"bench-alerts-{run_id}-{int(rate)}"
    n_messages = max(100, int(rate * duration_sec))

    ensure_topics(bootstrap_servers, [flows_topic, alerts_topic])

    consumer_result: dict = {}
    consumer_exc: list[Exception] = []

    def _run_consumer():
        try:
            res = consume_and_score(
                bootstrap_servers=bootstrap_servers,
                flows_topic=flows_topic,
                alerts_topic=alerts_topic,
                model_name=model_name,
                group_id=f"bench-grp-{run_id}-{int(rate)}",
                batch_size=batch_size,
                max_messages=n_messages,
                idle_timeout_sec=10.0,
                auto_offset_reset="earliest",
            )
            consumer_result.update(res)
        except Exception as exc:
            consumer_exc.append(exc)

    t_cons = threading.Thread(target=_run_consumer, daemon=True)
    t_cons.start()

    # Give consumer time to load model and complete group rebalance/partition assignment
    time.sleep(2.5)

    prod_summary = replay_flows(
        bootstrap_servers=bootstrap_servers,
        topic=flows_topic,
        rate=float(rate),
        max_messages=n_messages,
        shuffle=True,
        sample_seed=42,
        run_id=f"{run_id}-{int(rate)}",
    )

    t_cons.join(timeout=60)
    if consumer_exc:
        raise consumer_exc[0]

    consumed = consumer_result.get("consumed", 0)
    lag_at_end = max(0, prod_summary["sent"] - consumed)

    return {
        "target_rate_msgs_per_sec": rate,
        "planned_messages": n_messages,
        "produced_messages": prod_summary["sent"],
        "producer_achieved_rate": prod_summary["achieved_rate"],
        "producer_elapsed_sec": prod_summary["elapsed_sec"],
        "consumed_messages": consumed,
        "consumer_achieved_throughput": consumer_result.get("throughput_msgs_per_sec", 0.0),
        "consumer_elapsed_sec": consumer_result.get("elapsed_sec", 0.0),
        "alerts_emitted": consumer_result.get("alerts_emitted", 0),
        "latency_ms": consumer_result.get("latency_ms", {}),
        "consumer_lag_at_end": lag_at_end,
        "online_quality": consumer_result.get("online_quality", {}),
    }


def run_benchmark_suite(
    rates: list[float] = DEFAULT_RATES,
    duration_sec: float = 8.0,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    model_name: str = "xgboost_baseline",
    batch_size: int = 64,
    output_path: str = DEFAULT_OUTPUT,
) -> dict:
    """Run benchmarks across all configured rates and save results/benchmark.json."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    specs = get_machine_specs()
    results_per_rate = []

    print(f"Starting StreamGuard-5G Benchmark (run_id={run_id}, model={model_name}, batch_size={batch_size})")
    print(f"Machine: {specs['cpu_model']} | Cores: {specs['cpu_cores_logical']} | RAM: {specs['ram_gb']} GB")

    for rate in rates:
        print(f"\n--- Benchmarking target rate: {rate} msgs/sec ({duration_sec}s) ---")
        res = run_single_rate_benchmark(
            rate=rate,
            duration_sec=duration_sec,
            bootstrap_servers=bootstrap_servers,
            model_name=model_name,
            batch_size=batch_size,
            run_id=run_id,
        )
        results_per_rate.append(res)
        lat = res["latency_ms"]
        print(
            f"Produced: {res['produced_messages']} ({res['producer_achieved_rate']} msg/s) | "
            f"Consumed: {res['consumed_messages']} ({res['consumer_achieved_throughput']} msg/s) | "
            f"Latency p50={lat.get('p50')}ms p95={lat.get('p95')}ms p99={lat.get('p99')}ms | "
            f"Lag={res['consumer_lag_at_end']}"
        )

    payload = {
        "benchmark_timestamp": time.time(),
        "run_id": run_id,
        "model": model_name,
        "batch_size": batch_size,
        "duration_per_rate_sec": duration_sec,
        "machine_specs": specs,
        "results": results_per_rate,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved benchmark results to {output_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="StreamGuard-5G Streaming Benchmark")
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--model", default="xgboost_baseline", choices=["xgboost_baseline", "isolation_forest"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_benchmark_suite(
        rates=args.rates,
        duration_sec=args.duration_sec,
        bootstrap_servers=args.bootstrap_servers,
        model_name=args.model,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
