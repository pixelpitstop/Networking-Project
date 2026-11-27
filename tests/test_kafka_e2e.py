"""
End-to-end Kafka streaming pipeline test (producer -> flows -> consumer -> alerts).
Requires a running Kafka broker on localhost:9092.
"""

import json
import uuid
import threading
import pytest
from confluent_kafka import Consumer, KafkaError

from src.producer import replay_flows
from src.consumer import consume_and_score

BOOTSTRAP_SERVERS = "localhost:9092"


def test_end_to_end_kafka_replay_and_alerting():
    run_id = uuid.uuid4().hex[:8]
    flows_topic = f"flows-e2e-{run_id}"
    alerts_topic = f"alerts-e2e-{run_id}"
    n_messages = 2000

    # Replay 2,000 stratified test flows at 1,000 msgs/sec
    prod_summary = replay_flows(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        topic=flows_topic,
        rate=1000.0,
        max_messages=n_messages,
        shuffle=True,
        sample_seed=42,
        run_id=run_id,
    )
    assert prod_summary["sent"] == n_messages
    assert prod_summary["errors"] == 0

    # Consume, score with XGBoost, and publish alerts
    cons_summary = consume_and_score(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        flows_topic=flows_topic,
        alerts_topic=alerts_topic,
        model_name="xgboost_baseline",
        group_id=f"e2e-group-{run_id}",
        batch_size=64,
        max_messages=n_messages,
        idle_timeout_sec=5.0,
        auto_offset_reset="earliest",
    )
    assert cons_summary["consumed"] == n_messages
    assert cons_summary["alerts_emitted"] > 0

    # Offline evaluation alert rate is ~86.2% (445,789 / 517,156); expect roughly similar online
    alert_rate = cons_summary["alert_rate"]
    assert 0.75 <= alert_rate <= 0.95, f"Unexpected online alert rate: {alert_rate}"
    assert cons_summary["online_quality"]["precision"] >= 0.98
    assert cons_summary["online_quality"]["recall"] >= 0.90

    # Verify alerts can be read back from the alerts topic with required fields
    alert_reader = Consumer({
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "group.id": f"e2e-alert-reader-{run_id}",
        "auto.offset.reset": "earliest",
    })
    alert_reader.subscribe([alerts_topic])

    sample_alerts = []
    try:
        for _ in range(30):
            msg = alert_reader.poll(timeout=0.5)
            if msg is None:
                if sample_alerts:
                    break
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise RuntimeError(msg.error())
            sample_alerts.append(json.loads(msg.value().decode("utf-8")))
            if len(sample_alerts) >= 5:
                break
    finally:
        alert_reader.close()

    assert len(sample_alerts) > 0
    for alert in sample_alerts:
        assert alert["predicted_label"] == 1
        assert "score" in alert
        assert "produce_ts" in alert
        assert "consume_ts" in alert
        assert alert["consume_ts"] >= alert["produce_ts"]
