import time
from kafka.structs import TopicPartition
from kafka import KafkaConsumer

# class MultiThreadKafka(object):
#     # 消费者端多线程Kafka
#     def __init__(self, max_workers=2):
#         try:
#             self.tp = TopicPartition("test", 0)  # (topic, partition)
#             self.consumer = KafkaConsumer(bootstrap_servers='139.159.179.226:9092')
#             self.consumer.assign([self.tp])
#             offset_dict = self.consumer.end_offsets([self.tp])
#             self.seek = list(offset_dict.values())[0]  # 默认获取最新的offset
#             self.max_workers = max_workers
#         except Exception as e:
#             print(e)

# if __name__ == '__main__':
#     Kafka = MultiThreadKafka(max_workers=2)
#     for msg in Kafka.consumer:
#         print("%s:%d:%d: key=%s value=%s" %
#             (msg.topic, msg.partition, msg.offset, msg.key, msg.value))
#         print()


consumer = KafkaConsumer(
    'test',
    group_id='test_id',
    bootstrap_servers=['139.159.179.226:9092'],  # 要接收的kafka主题
    auto_offset_reset='earliest')

# consumer = KafkaConsumer(
#     'test',
#     group_id='test_id',
#     bootstrap_servers=['127.0.0.1:9092'],  # 要接收的kafka主题
#     auto_offset_reset='earliest')
for msg in consumer:
    print(msg)
