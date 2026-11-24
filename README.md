# StreamGuard-5G

Real-time anomaly detector for 5G network traffic.

## Project Overview

StreamGuard-5G trains machine learning anomaly detection models on 5G network flow records and radio telemetry (NetsLab 5G Open RAN IDD dataset), replays flows through an Apache Kafka streaming pipeline, and scores incoming flows in real time to publish alerts while benchmarking throughput and latency.

> **Key Limitation:** The stream is a replay of recorded testbed data, not live live-traffic tap on a carrier network.

## Architecture

```
[Flow Replayer (producer.py)] ---> Kafka Topic: 'flows' ---> [Detector (consumer.py)] ---> Kafka Topic: 'alerts'
                                                                     |
                                                              [Scoring Model]
                                                          (Isolation Forest / XGBoost)
```

## Setup & Quickstart

### Prerequisites
- Python 3.11+
- Docker and Docker Compose

### 1. Environment Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Kafka Broker
```bash
docker-compose up -d
```

### 3. Verification
```bash
python -m pytest tests/test_kafka_smoke.py
```
