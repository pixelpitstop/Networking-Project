# StreamGuard-5G: Project Brief

> Instructions for the coding agent: read this whole file first. Work **one phase at a time**. At the end of each phase, stop, summarise what you did and what the measured results are, and wait for the human to say "continue". Never start the next phase on your own.

## 1. Goal

Build a real-time anomaly detector for 5G network traffic.

- Train models on labeled 5G network flows (5G-NIDD dataset).
- Replay the flows through Kafka as a stream.
- Score each flow as it arrives, publish alerts, and measure both **detection quality** and **streaming performance** (throughput, latency).

Audience: an interviewer at a telecom R&D lab. The repo must be honest, reproducible and explainable, not flashy.

## 2. Hard rules (do not break these)

1. **No invented numbers.** Every metric, throughput or latency figure in the README must come from a file produced by the code (`results/*.json`). The README tables must be generated from those files by a script, never typed by hand.
2. **Do not assume the dataset schema.** Inspect the real columns first (Phase 1) and write down what you found in `docs/DATA_NOTES.md`.
3. **No label leakage.** Remove identifier or leaky columns (IP addresses, timestamps, sequence numbers, anything that identifies an attack source) and document every dropped column and why.
4. **State the limitation plainly:** the stream is a *replay of recorded data*, not live 5G traffic.
5. Pin dependency versions, set random seeds, commit after each phase with a clear message.
6. Keep scope small. Do **not** add deep learning, cloud deployment, auth or extra services unless the human asks.

## 3. Stack

- Python 3.11, pandas, numpy, scikit-learn, xgboost, joblib, matplotlib
- Kafka via Docker Compose (single-broker, KRaft or Redpanda), client: `confluent-kafka`
- pytest for tests
- Streamlit for a small alerts view (Phase 6, optional)

## 4. Data

**Primary:** 5G-NIDD, a labeled intrusion-detection dataset collected on the 5G Test Network at the University of Oulu. It has benign traffic plus DoS attacks (ICMP, UDP, SYN, HTTP flood, slow-rate DoS) and port scans (SYN, TCP connect, UDP). It is published on IEEE DataPort (DOI 10.21227/xtep-hv36) and mirrored on a Finnish repository.

**Fallback:** NetsLab 5G Open RAN Intrusion Detection Dataset on Kaggle (network flows plus radio-level metrics).

**The human downloads the data manually** (the IEEE DataPort download needs a free login) and places it in `data/raw/`. Do not try to scrape or bypass logins. If the files are missing, stop and tell the human exactly what to put where. Never commit raw data to git; add `data/` to `.gitignore` and document the download steps in the README.

## 5. Repository layout

```
streamguard-5g/
  BRIEF.md
  README.md
  docker-compose.yml
  requirements.txt
  data/            # gitignored: raw/ and processed/
  docs/
    DATA_NOTES.md  # columns, label meaning, dropped columns and why
    EXPLAINER.md   # plain-English concepts (see Phase 8)
  src/
    data.py        # load, clean, split
    features.py
    train.py       # trains both model tracks, saves to models/
    evaluate.py    # metrics + per-attack recall -> results/metrics.json
    producer.py    # replays rows into Kafka at a given rate
    consumer.py    # scores flows, writes alerts topic
    benchmark.py   # throughput and latency -> results/benchmark.json
    report.py      # builds README tables from results/*.json
  tests/
  models/
  results/
```

## 6. Phases and acceptance criteria

### Phase 0: Setup
- Create the repo layout, `requirements.txt`, `.gitignore`, a minimal README stub, and `docker-compose.yml` for Kafka.
- **Done when:** `docker compose up` starts the broker, and a tiny script can produce and consume one test message.

### Phase 1: Data understanding
- Load the data, list columns and types, label distribution (benign vs attack, per attack type), missing values, duplicates.
- Write `docs/DATA_NOTES.md` including which columns are leakage risks and which you will drop.
- **Done when:** the notes file exists and a label-distribution table is saved to `results/label_distribution.json`.

### Phase 2: Split and baseline models
- **Preferred split:** train on one base station's data, test on the other's (the dataset provides per-base-station files), to test generalisation. If that is not possible, use a time-ordered split. Never use a random shuffle split without explaining the leakage risk. Document the choice.
- Track A (unsupervised): Isolation Forest trained on **benign traffic only**.
- Track B (supervised baseline): XGBoost (or Random Forest) with class weighting for the imbalance.
- **Done when:** both models are saved in `models/` and `train.py` reproduces them from a clean checkout.

### Phase 3: Evaluation
- Report precision, recall, F1, PR-AUC and false-positive rate for both tracks on the test split.
- Report **recall per attack type** (so we can see which attacks are easy and which are hard, e.g. floods vs scans vs slow-rate DoS).
- Include a short confusion-matrix plot and a PR curve plot in `results/`.
- **Done when:** `results/metrics.json` and plots exist. Add a short, honest written finding in `docs/FINDINGS.md` about where each model fails and why you think so (mark hypotheses as hypotheses).

### Phase 4: Streaming pipeline
- `producer.py` replays test rows into topic `flows` at a configurable rate (messages/sec), embedding a produce timestamp in each message.
- `consumer.py` consumes from `flows`, scores with the chosen model, and writes anomalies to topic `alerts` (include score, predicted label, produce timestamp, consume timestamp).
- **Done when:** end-to-end replay works, alert counts roughly agree with the offline evaluation, and a test covers the scoring function.

### Phase 5: Benchmark
- Run replay at several rates (for example 100, 500, 1000, 2000 msgs/sec, adjust to what the machine can sustain).
- Record per rate: achieved throughput, end-to-end latency p50/p95/p99, and consumer lag at the end.
- Save to `results/benchmark.json` and note the machine specs (CPU, RAM, single broker) in the same file.
- **Done when:** the benchmark is reproducible with one documented command.

### Phase 6 (optional): Alerts dashboard
- Small Streamlit page reading the `alerts` topic: alert rate over time, top predicted attack types. Keep it under about 150 lines.

### Phase 7: README and report
- `report.py` generates the results tables into the README from `results/*.json`.
- README sections: what it is, architecture diagram (simple, text or image), dataset and citation, how to reproduce, results tables, **limitations**, future work.
- Cite the dataset authors per the dataset page.

### Phase 8: Explainer for the human
- Write `docs/EXPLAINER.md` in plain English covering: what a network flow is; how a SYN flood, UDP flood, slow-rate DoS and port scan differ and how they look in the features; why accuracy misleads on imbalanced data; why Isolation Forest is trained on benign data only; what a false positive costs a network operator; Kafka topics, partitions, consumer groups and offsets; why the split was chosen; every known limitation.
- End with 15 likely interview questions about this project and short model answers grounded in **this repo's actual results**.

## 7. Definition of done

- Fresh clone runs end to end with the documented commands.
- All numbers in the README are generated from result files.
- Tests pass.
- Limitations are stated honestly.
- The human can explain every file in `src/` in two minutes.