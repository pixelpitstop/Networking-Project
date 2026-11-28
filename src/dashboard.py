"""
StreamGuard-5G Alerts Dashboard (Streamlit).
Reads anomaly alerts from the Kafka `alerts` topic and displays alert rate over time
and top predicted attack types.

Run with:
    streamlit run src/dashboard.py
"""

import json
import uuid
import pandas as pd
import streamlit as st
from confluent_kafka import Consumer, KafkaError


def poll_alerts(
    bootstrap_servers: str = "localhost:9092",
    topic: str = "alerts",
    max_records: int = 5000,
    timeout_sec: float = 1.5,
) -> list[dict]:
    """Read available alert records from the Kafka alerts topic."""
    consumer = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": f"streamguard-dashboard-{uuid.uuid4().hex[:8]}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([topic])
    records: list[dict] = []
    try:
        msgs = consumer.consume(num_messages=max_records, timeout=timeout_sec)
        for msg in msgs:
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                continue
            records.append(json.loads(msg.value().decode("utf-8")))
    finally:
        consumer.close()
    return records


def build_alert_rate_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate alert counts per second from consume_ts."""
    if df.empty or "consume_ts" not in df.columns:
        return pd.DataFrame(columns=["alerts_per_sec"])
    ts = pd.to_datetime(df["consume_ts"], unit="s", errors="coerce").dropna()
    if ts.empty:
        return pd.DataFrame(columns=["alerts_per_sec"])
    counts = ts.dt.floor("1s").value_counts().sort_index()
    return counts.rename("alerts_per_sec").to_frame()


def top_attack_types(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Return top attack types among emitted alerts."""
    if df.empty:
        return pd.Series(dtype=int)
    col = "attack_type" if "attack_type" in df.columns else "model"
    return df[col].fillna("unknown").value_counts().head(top_n)


def main():
    st.set_page_config(page_title="StreamGuard-5G Alerts", layout="wide")
    st.title("StreamGuard-5G: Real-Time Anomaly Alerts")
    st.caption("Note: Stream is a replay of recorded testbed flows (NetsLab 5G O-RAN IDD), not live 5G traffic.")

    with st.sidebar:
        bootstrap = st.text_input("Kafka Bootstrap", value="localhost:9092")
        topic = st.text_input("Alerts Topic", value="alerts")
        max_records = st.slider("Max Alerts to Load", 100, 20000, 5000, step=500)
        if st.button("Refresh Alerts"):
            st.rerun()

    records = poll_alerts(bootstrap_servers=bootstrap, topic=topic, max_records=max_records)
    if not records:
        st.info(f"No alerts found on topic `{topic}`. Run `src/producer.py` and `src/consumer.py` to stream flows.")
        return

    df = pd.DataFrame(records)
    c1, c2, c3 = st.columns(3)
    c1.metric("Alerts Loaded", f"{len(df):,}")
    c2.metric("Mean Anomaly Score", f"{df['score'].mean():.4f}" if "score" in df else "N/A")
    c3.metric("Median Latency (ms)", f"{df['latency_ms'].median():.2f}" if "latency_ms" in df else "N/A")

    left, right = st.columns(2)
    with left:
        st.subheader("Alert Rate Over Time (alerts/sec)")
        rate_df = build_alert_rate_series(df)
        if not rate_df.empty:
            st.line_chart(rate_df)
    with right:
        st.subheader("Top Predicted Attack Types")
        top_df = top_attack_types(df)
        if not top_df.empty:
            st.bar_chart(top_df)

    st.subheader("Recent Alerts")
    cols = [c for c in ["flow_id", "model", "score", "attack_category", "attack_type", "latency_ms"] if c in df.columns]
    st.dataframe(df[cols].tail(20), use_container_width=True)


if __name__ == "__main__":
    main()
