import time
import json
import uuid
import pytest
from confluent_kafka import Producer, Consumer, KafkaError

BOOTSTRAP_SERVERS = "localhost:9092"
TEST_TOPIC = "streamguard-test-topic"

def test_kafka_produce_consume():
    test_id = str(uuid.uuid4())
    payload = {"test_id": test_id, "timestamp": time.time(), "message": "hello streamguard"}

    # Producer setup
    producer_conf = {
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "client.id": "test-smoke-producer"
    }
    producer = Producer(producer_conf)

    delivery_ack = []
    def on_delivery(err, msg):
        if err:
            delivery_ack.append(False)
        else:
            delivery_ack.append(True)

    producer.produce(
        TEST_TOPIC,
        key=test_id.encode("utf-8"),
        value=json.dumps(payload).encode("utf-8"),
        callback=on_delivery
    )
    producer.flush(timeout=10)
    assert len(delivery_ack) > 0 and delivery_ack[0], "Failed to deliver message to Kafka"

    # Consumer setup
    consumer_conf = {
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "group.id": f"smoke-test-group-{uuid.uuid4()}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([TEST_TOPIC])

    received = None
    start_time = time.time()
    try:
        while time.time() - start_time < 15:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    raise Exception(msg.error())
            
            data = json.loads(msg.value().decode("utf-8"))
            if data.get("test_id") == test_id:
                received = data
                break
    finally:
        consumer.close()

    assert received is not None, f"Did not receive message with test_id={test_id} within timeout"
    assert received["message"] == "hello streamguard"
    print("\n[SUCCESS] Successfully produced and consumed test message through Kafka broker!")

if __name__ == "__main__":
    test_kafka_produce_consume()
