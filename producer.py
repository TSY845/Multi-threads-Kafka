from kafka import KafkaProducer


producer = KafkaProducer(bootstrap_servers=["127.0.0.1:9092"])
 
for i in range(10):
    s = producer.send(topic="test", value=b'Hello Kafka. This is a test message.', partition=0)

