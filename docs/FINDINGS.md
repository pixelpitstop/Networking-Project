# Model Evaluation Findings & Error Analysis

This document summarizes the offline evaluation results on the held-out test split (`517,156` flows: `51,260` benign, `465,896` attack) from [`results/metrics.json`](../results/metrics.json).

---

## 1. Overall Comparison

| Metric | Track A: Isolation Forest (Unsupervised) | Track B: XGBoost Baseline (Supervised) |
| :--- | :---: | :---: |
| **Precision** | 0.7441 | 0.9980 |
| **Recall** | 0.0209 | 0.9550 |
| **F1 Score** | 0.0406 | 0.9760 |
| **PR-AUC** | 0.9101 | 0.9991 |
| **ROC-AUC** | 0.6099 | 0.9923 |
| **False-Positive Rate (FPR)** | 0.0652 (3,343 FP) | 0.0170 (872 FP) |
| **Scoring Time (517k flows)** | 1.80 s | 0.15 s |

> **Important Note on PR-AUC vs. Prevalence:** Because the test split is 90.09% attack traffic (`0.9009` positive prevalence), a no-skill classifier achieves a PR-AUC of `0.9009`. Track A's `0.9101` PR-AUC is therefore only marginally above chance, which is confirmed by its `0.6099` ROC-AUC and `0.0209` recall.

---

## 2. Where Track A (Isolation Forest) Fails

Track A flags only **9,719 of 465,896** attacks (`2.09%` recall) while generating **3,343 false positives** (`6.52%` FPR) on benign traffic:
- **Zero or near-zero recall on floods & scans:** `http_flood` (`0.0000`), `portscan_tcp` (`0.0000`), `portscan_udp` (`0.0000`), `dir_bruteforce` (`0.0000`), `syn` (`0.0002`), and `tcp_ack` (`0.0002`).
- **Slightly better only on `udp` DoS (`0.1386`) and `ssh` (`0.0212`).**

### Hypotheses for Track A Failure
- **[Hypothesis 1 — Benign Diversity vs. Dense Attack Clusters]:** Benign traffic spans diverse applications (`http`, `ssl`, `dns`, `quic`, file transfers, varying durations and byte sizes). By contrast, floods (`syn`, `http_flood`) and port scans consist of thousands of near-identical, minimal-payload packets. In an axis-aligned random tree ensemble, sparse outlier benign flows (e.g., long-lived file downloads) sit farther from the core than dense, repetitive small-packet floods, causing `IsolationForest` to isolate normal edge cases first.
- **[Hypothesis 2 — Curse of High-Dimensional One-Hot Encoding]:** Our feature pipeline expands 3 categoricals (`proto`, `service`, `conn_state`) into 31 one-hot binary columns out of 51 total features. Standard Isolation Forest treats all 51 dimensions with equal probability when selecting split features, diluting the signal of critical continuous rate features (`pkt_rate`, `byte_rate`, `duration`).
- **[Hypothesis 3 — Conservative `contamination=0.01` Threshold & Benign Drift]:** Setting `contamination=0.01` places the decision boundary at the extreme 1% tail of the benign training set. Moreover, the trailing 30% benign test split exhibits a `6.52%` FPR (up from `1.0%` on train), indicating temporal drift between earlier and later benign capture sessions.

---

## 3. Where Track B (XGBoost Baseline) Succeeds and Fails

Track B achieves strong overall performance (`F1 = 0.9760`, `FPR = 0.0170`), but per-attack-type evaluation exposes clear blind spots:

- **High Recall (>96%):** `http_flood` (`0.9999`), `dir_bruteforce` (`0.9992`), `portscan_tcp` (`0.9983`), `portscan_udp` (`0.9980`), `syn` (`0.9913`), `tcp_ack` (`0.9657`).
- **Moderate Recall (62%–93%):** `udp` (`0.9273`), `os_fingerprinting` (`0.9221`), `slowloris` (`0.8799`), `icmp` (`0.6924`), `ftp` (`0.6220`).
- **Severe Blind Spots (<45%):**
  - `ssh` brute force: **`0.4440`** (2,486 missed out of 4,471)
  - `xss`: **`0.1756`** (906 missed out of 1,099)
  - `sql_injection`: **`0.1474`** (451 missed out of 529)

### Hypotheses for Track B Failure Cases
- **[Hypothesis 4 — Lack of Application-Layer Payload Visibility]:** `sql_injection` (`14.74%` recall) and `xss` (`17.56%` recall) are carried inside completely valid HTTP TCP connections (`conn_state=SF`, `service=http`). Once leaky testbed IPs (`src_ip`, `dst_ip`) are removed, their Layer 3/4 flow metadata (packet count, byte size, duration) closely overlaps with benign web requests. Without Deep Packet Inspection (DPI) of HTTP URIs/payloads, flow-level models cannot reliably separate malicious payloads from normal HTTP GET/POST traffic.
- **[Hypothesis 5 — Extreme Sub-Class Rarity & Class Weighting]:** In the test split, `sql_injection` (`529` flows) and `xss` (`1,099` flows) together represent `<0.32%` of all test records. Because binary `scale_pos_weight = 0.1100` down-weights all attack samples uniformly, rare attack sub-types contribute negligible gradient compared to massive flood classes (`syn` = `151,937`, `http_flood` = `76,295`).
- **[Hypothesis 6 — Single-Flow vs. Windowed Context for Brute Force & Slow-Rate DoS]:** `ssh` (`44.40%`), `ftp` (`62.20%`), and `slowloris` (`87.99%`) exhibit lower recall than volumetric floods because individual connections look structurally similar to legitimate sessions. Detecting credential stuffing (`ssh`/`ftp`) or slow-rate exhaustion (`slowloris`) reliably requires **temporal/host-windowed aggregation features** (e.g., failed connections per destination port in a 10-second sliding window), which single-flow scoring lacks.

---

## 4. Summary Takeaway for Streaming (Phase 4)

Track B (`xgboost_baseline`) is the clear choice for the live Kafka consumer in Phase 4 due to its `0.9760` F1, `1.70%` false-positive rate, and `0.15s` batch scoring speed across 517k flows (~3.4M flows/sec offline). However, operators must be aware that **single-flow metadata scoring will miss most SQLi/XSS payload attacks and ~55% of SSH brute-force flows**.
