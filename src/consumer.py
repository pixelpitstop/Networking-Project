"""
Kafka scoring consumer (consumer.py).
Consumes flows from Kafka topic `flows`, scores them with the trained ML model,
and publishes detected anomalies to topic `alerts`.
"""

import os
import sys
import json
import time
import uuid
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from confluent_kafka import Consumer, Producer, KafkaError

DEFAULT_BOOTSTRAP = "localhost:9092"
DEFAULT_FLOWS_TOPIC = "flows"
DEFAULT_ALERTS_TOPIC = "alerts"
DEFAULT_MODELS_DIR = "models"


class FlowScorer:
    """Loads the feature pipeline and trained model to score incoming flow records."""

    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR, model_name: str = "xgboost_baseline"):
        if model_name not in ("xgboost_baseline", "isolation_forest"):
            raise ValueError(f"Unsupported model_name: {model_name}. Choose 'xgboost_baseline' or 'isolation_forest'.")

        self.model_name = model_name
        self.pipeline = joblib.load(os.path.join(models_dir, "feature_pipeline.joblib"))
        self.model = joblib.load(os.path.join(models_dir, f"{model_name}.joblib"))

    def score_batch(self, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """
        Score a batch of raw flow dictionaries.
        Returns (predicted_labels, anomaly_scores) where predicted_label=1 indicates anomaly.
        """
        if not records:
            return np.array([], dtype=int), np.array([], dtype=float)

        df = pd.DataFrame(records)
        X = self.pipeline.transform(df)

        if self.model_name == "xgboost_baseline":
            preds = self.model.predict(X).astype(int)
            scores = self.model.predict_proba(X)[:, 1].astype(float)
        else:
            preds = (self.model.predict(X) == -1).astype(int)
            scores = (-self.model.score_samples(X)).astype(float)

        return preds, scores

    def score_one(self, record: dict) -> tuple[int, float]:
        """Score a single flow dictionary."""
        preds, scores = self.score_batch([record])
        return int(preds[0]), float(scores[0])


def consume_and_score(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    flows_topic: str = DEFAULT_FLOWS_TOPIC,
    alerts_topic: str = DEFAULT_ALERTS_TOPIC,
    models_dir: str = DEFAULT_MODELS_DIR,
    model_name: str = "xgboost_baseline",
    group_id: str | None = None,
    batch_size: int = 64,
    max_messages: int | None = None,
    idle_timeout_sec: float = 5.0,
    auto_offset_reset: str = "earliest",
) -> dict:
    """
    Consume flow messages from `flows_topic`, score them, and emit anomalies to `alerts_topic`.
    Stops when `max_messages` are processed or `idle_timeout_sec` elapses with no new messages.
    """
    scorer = FlowScorer(models_dir=models_dir, model_name=model_name)
    group_id = group_id or f"streamguard-consumer-{uuid.uuid4().hex[:8]}"

    consumer = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": auto_offset_reset,
        "enable.auto.commit": True,
    })
    producer = Producer({
        "bootstrap.servers": bootstrap_servers,
        "client.id": f"streamguard-alerts-{group_id}",
        "linger.ms": 5,
    })

    consumer.subscribe([flows_topic])

    consumed = 0
    alerts_emitted = 0
    tp = fp = fn = tn = 0
    latencies_ms: list[float] = []
    start_wall: float | None = None
    last_msg_wall = time.perf_counter()

    try:
        while True:
            if max_messages is not None and consumed >= max_messages:
                break

            to_read = batch_size
            if max_messages is not None:
                to_read = min(batch_size, max_messages - consumed)

            msgs = consumer.consume(num_messages=to_read, timeout=0.2)
            now_wall = time.perf_counter()

            if not msgs:
                if now_wall - last_msg_wall >= idle_timeout_sec:
                    break
                continue

            records = []
            for msg in msgs:
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise RuntimeError(f"Kafka consumer error: {msg.error()}")
                records.append(json.loads(msg.value().decode("utf-8")))

            if not records:
                if now_wall - last_msg_wall >= idle_timeout_sec:
                    break
                continue

            if start_wall is None:
                start_wall = time.perf_counter()
            last_msg_wall = time.perf_counter()

            preds, scores = scorer.score_batch(records)
            consume_ts = time.time()

            for rec, pred, score in zip(records, preds, scores):
                consumed += 1
                pred_int = int(pred)
                score_float = float(score)
                produce_ts = rec.get("produce_ts")
                if produce_ts is not None:
                    latencies_ms.append((consume_ts - float(produce_ts)) * 1000.0)

                true_label = rec.get("traffic_type")
                if true_label is not None:
                    true_int = int(true_label)
                    if pred_int == 1 and true_int == 1:
                        tp += 1
                    elif pred_int == 1 and true_int == 0:
                        fp += 1
                    elif pred_int == 0 and true_int == 1:
                        fn += 1
                    else:
                        tn += 1

                if pred_int == 1:
                    alerts_emitted += 1
                    alert_payload = {
                        "flow_id": rec.get("flow_id"),
                        "predicted_label": pred_int,
                        "score": round(score_float, 6),
                        "model": model_name,
                        "produce_ts": produce_ts,
                        "consume_ts": consume_ts,
                        "latency_ms": round((consume_ts - float(produce_ts)) * 1000.0, 3) if produce_ts is not None else None,
                        "attack_category": rec.get("attack_category"),
                        "attack_type": rec.get("attack_type"),
                        "true_label": true_label,
                    }
                    producer.produce(
                        alerts_topic,
                        key=str(rec.get("flow_id", "")).encode("utf-8"),
                        value=json.dumps(alert_payload).encode("utf-8"),
                    )

            producer.poll(0)

    finally:
        producer.flush(timeout=15)
        consumer.close()

    elapsed = (time.perf_counter() - start_wall) if start_wall is not None else 0.0
    throughput = consumed / elapsed if elapsed > 0 else 0.0

    lat_arr = np.array(latencies_ms) if latencies_ms else np.array([0.0])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "model": model_name,
        "flows_topic": flows_topic,
        "alerts_topic": alerts_topic,
        "consumed": consumed,
        "alerts_emitted": alerts_emitted,
        "alert_rate": round(alerts_emitted / consumed, 4) if consumed > 0 else 0.0,
        "elapsed_sec": round(elapsed, 3),
        "throughput_msgs_per_sec": round(throughput, 2),
        "latency_ms": {
            "p50": round(float(np.percentile(lat_arr, 50)), 2),
            "p95": round(float(np.percentile(lat_arr, 95)), 2),
            "p99": round(float(np.percentile(lat_arr, 99)), 2),
        },
        "online_quality": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="StreamGuard-5G Kafka Scoring Consumer")
    parser.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--flows-topic", default=DEFAULT_FLOWS_TOPIC)
    parser.add_argument("--alerts-topic", default=DEFAULT_ALERTS_TOPIC)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model", default="xgboost_baseline", choices=["xgboost_baseline", "isolation_forest"])
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--idle-timeout", type=float, default=5.0)
    parser.add_argument("--offset-reset", default="earliest", choices=["earliest", "latest"])
    args = parser.parse_args()

    summary = consume_and_score(
        bootstrap_servers=args.bootstrap_servers,
        flows_topic=args.flows_topic,
        alerts_topic=args.alerts_topic,
        models_dir=args.models_dir,
        model_name=args.model,
        group_id=args.group_id,
        batch_size=args.batch_size,
        max_messages=args.max_messages,
        idle_timeout_sec=args.idle_timeout,
        auto_offset_reset=args.offset_reset,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
