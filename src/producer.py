"""
Kafka flow replayer (producer.py).
Replays recorded 5G test flows into the Kafka `flows` topic at a configurable rate,
embedding a produce timestamp in each message.

Limitation: This replays recorded testbed data, not live 5G network traffic.
"""

import os
import sys
import json
import time
import uuid
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from confluent_kafka import Producer

DEFAULT_TEST_PATH = "data/processed/test.parquet"
DEFAULT_BOOTSTRAP = "localhost:9092"
DEFAULT_TOPIC = "flows"


def load_replay_flows(
    test_path: str = DEFAULT_TEST_PATH,
    max_messages: int | None = None,
    shuffle: bool = True,
    sample_seed: int = 42,
) -> list[dict]:
    """Load test flows as a list of JSON-serializable dicts."""
    df = pd.read_parquet(test_path)
    if shuffle:
        if max_messages is not None and max_messages < len(df):
            df = df.sample(n=max_messages, random_state=sample_seed)
        else:
            df = df.sample(frac=1.0, random_state=sample_seed)
    elif max_messages is not None and max_messages < len(df):
        df = df.iloc[:max_messages]

    return df.to_dict(orient="records")


def replay_flows(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    topic: str = DEFAULT_TOPIC,
    test_path: str = DEFAULT_TEST_PATH,
    rate: float = 500.0,
    max_messages: int | None = None,
    shuffle: bool = True,
    sample_seed: int = 42,
    run_id: str | None = None,
) -> dict:
    """
    Replay test flows into Kafka topic at `rate` messages/sec (0 = unthrottled).
    Returns summary statistics for the replay run.
    """
    run_id = run_id or str(uuid.uuid4())[:8]
    records = load_replay_flows(
        test_path=test_path,
        max_messages=max_messages,
        shuffle=shuffle,
        sample_seed=sample_seed,
    )

    producer = Producer({
        "bootstrap.servers": bootstrap_servers,
        "client.id": f"streamguard-producer-{run_id}",
        "linger.ms": 5,
        "batch.num.messages": 1000,
        "queue.buffering.max.messages": 200000,
    })

    delivery_errors = 0

    def _on_delivery(err, _msg):
        nonlocal delivery_errors
        if err is not None:
            delivery_errors += 1

    total = len(records)
    start_time = time.perf_counter()

    for idx, rec in enumerate(records):
        payload = dict(rec)
        payload["flow_id"] = f"{run_id}-{idx}"
        payload["produce_ts"] = time.time()
        encoded = json.dumps(payload).encode("utf-8")

        while True:
            try:
                producer.produce(
                    topic,
                    key=payload["flow_id"].encode("utf-8"),
                    value=encoded,
                    callback=_on_delivery,
                )
                break
            except BufferError:
                producer.poll(0.01)

        if idx % 100 == 0:
            producer.poll(0)

        if rate and rate > 0:
            target_elapsed = (idx + 1) / rate
            actual_elapsed = time.perf_counter() - start_time
            sleep_for = target_elapsed - actual_elapsed
            if sleep_for > 0.0005:
                time.sleep(sleep_for)

    producer.flush(timeout=30)
    elapsed = time.perf_counter() - start_time
    achieved_rate = total / elapsed if elapsed > 0 else 0.0

    summary = {
        "run_id": run_id,
        "topic": topic,
        "sent": total - delivery_errors,
        "errors": delivery_errors,
        "target_rate": rate,
        "elapsed_sec": round(elapsed, 3),
        "achieved_rate": round(achieved_rate, 2),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="StreamGuard-5G Kafka Flow Producer")
    parser.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--rate", type=float, default=500.0, help="Messages/sec (0 for max speed)")
    parser.add_argument("--max-messages", type=int, default=None, help="Limit number of replayed flows")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable stratified random sampling/shuffling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    summary = replay_flows(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        test_path=args.test_path,
        rate=args.rate,
        max_messages=args.max_messages,
        shuffle=not args.no_shuffle,
        sample_seed=args.seed,
        run_id=args.run_id,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
