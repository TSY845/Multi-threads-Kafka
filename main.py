# encoding:utf-8
from kafka import KafkaProducer
from kafka import KafkaConsumer
import json
import time
import random

# 创建KafkaProducer连接实例
# producer = KafkaProducer(bootstrap_servers=["192.168.11.128:9092"])
# 将信息推送到主题（topic）中，推送的消息（message）最好转成json格式，或者bytes类型
# producer.send(topic, data)

'''
# 示例代码
topic = "newOrder"
producer = KafkaProducer(bootstrap_servers=["192.168.11.128:9092"])

for i in range(30):
    time.sleep(i)
    username = ["jcTang", "libai", "tufu", "tumu"]
    phone = ["151***4481", "159***6629", "189***7891", "131***5681"]
    random_num = random.randint(0, 3)
    orderinfo = {
        "orderNo": "no" + str(time.time()).replace(".", "")[:10]
        "username": username[random_num],
        "phone": phone[random_num],
        "goodid": random.randint(1, 100),
        "price": random.randint(1, 100),
        "buyNum": random.randint(1, 20),
        "allPrice": random.randint(1, 10000),
        "remark": "测试kafka"
    }
    data = json.dumps(orderinfo, ensure_ascii=True).encode("utf-8")
    print(data)
    producer.send(topic, data)
'''

'''
# Consumer(消费者)
# kafka-python消费数据，需要导入：KafkaConsumer
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'test',
    group_id='test_id',
    bootstrap_servers=['139.159.179.226:9092'],  # 要发送的kafka主题
    # auto_offset_reset='earliest',  # 有两个参数值，earliest和latest，如果省略这个参数，那么默认就是latest
)
for msg in consumer:
    print(msg)
'''

'''
# 创建KafkaConsumer连接实例
consumer = KafkaConsumer(
    "test",
    bootstrap_servers=["139.159.179.226:9092"]
)
for i in consumer:
    print(i.value)
'''

'''
import requests
# 客户端直接利用文件传输 时间消耗少
url = 'https://resttest.concoai.com//docking/identifyEventImage'
datas = {'url': "https://file.test.concoai.com/image/000248ec0307d7d09942af00e3840851.jpg", 'robotNumber': "123",
         'result': "true", 'tenantId': "123", "points": "[[0, 0, 100, 100], [200, 200, 300, 300]]"}
# data = json.dumps(datas)
r = requests.post(url, json=datas)
print(r.text)
print(r.status_code)
'''

'''
import requests

url = 'https://resttest.concoai.com//docking/identifyEventImage'
myobj = {'somekey': 'somevalue'}

x = requests.post(url, json=myobj)

print(x.text)
print(x.status_code)
'''


'''
import time
import threading


# 串行的普通编程方式
def syn_method():
    print("Start")
    time.sleep(1)
    print("Visit website 1")
    time.sleep(1)
    print("Visit website 2")
    time.sleep(1)
    print("Visit website 3")
    print("End")


# 异步多线程方式
def asyn_method():
    print("Start")
    def visit1():
        time.sleep(10)
        print("Visit website 1")
    def visit2():
        time.sleep(10)
        print("Visit website 2")
    def visit3():
        time.sleep(10)
        print("Visit website 3")
    # 首先定义多线程
    th1 = threading.Thread(target=visit1, daemon=False)  # 设置傀儡线程
    th2 = threading.Thread(target=visit2, daemon=False)
    th3 = threading.Thread(target=visit3, daemon=False)
    th1.start()
    th2.start()
    th3.start()
    # 最后汇总，等待线程1完成
    
    # th1.join()
    # th2.join()
    # th3.join()
    
    print("End")

asyn_method()
# syn_method()
'''

'''
# 不加锁
import threading

n = 0


def add(num):  # 累加函数
    global n
    for _ in range(num):
        n += 1


def sub(num):  # 累减函数
    global n
    for _ in range(num):
        n -= 1


num = 10000000
th_add = threading.Thread(target=add, args=(num, ))  # 定义线程，把num传参进去
th_sub = threading.Thread(target=sub, args=(num, ))
th_add.start()  # 启动线程
th_sub.start()
th_add.join()
th_sub.join()
print("n=", n)
print("End")
'''
'''
# 加锁
import threading
lock = threading.Lock()  # 定义一把锁
n = 0
def add(num):
    global n
    for i in range(num):
        lock.acquire()
        try:
            n += 1
        finally:
            lock.release()
def sub(num):
    global n
    for _ in range(num):
        lock.acquire()
        try:
            n -= 1
        finally:
            lock.release()

num = 10000000
th_add = threading.Thread(target=add, args=(num, ))
th_sub = threading.Thread(target=sub, args=(num, ))
th_add.start()
th_sub.start()
th_add.join()
th_sub.join()
print("n=", n)
print("End")
'''
'''
# 条件锁
# 程序功能：生产者消费者问题
import threading

class Buffer:  # 定义缓冲区类
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size  # 缓冲区大小
        self.buffer = []  # 用一个列表模拟缓冲区
        lock = threading.RLock()
        self.has_lock = threading.Condition(lock)
        self.has_data = threading.Condition(lock)
    def put(self, data, blocking=True):
        with self.has_lock:
            if blocking:   # 有缓冲空间
                while len(self.buffer) >= self.buffer_size:
                    self.has_lock.wait()  # 阻塞has_lock
            else:
                if len(self.buffer) >= self.buffer_size:
                    return False
            self.buffer.append(data)
            self.has_data.notify_all()  # 激活条件锁
            return True
    def get(self, blocking = True):  # 取数据（消费者）
        with self.has_data:
            if blocking:
                while len(self.buffer) == 0:
                    self.has_data.wait()
            else:
                if len(self.buffer) == 0:
                    return False
            result = self.buffer[0]
            del self.buffer[0]
            self.has_lock.notify_all()
            return result

if __name__ == '__main__':
    num = 20
    buffer = Buffer(5)
    def produce(n:int):
        for i in range(num):
            data = "data_%d_%d" % (n, i)
            buffer.put(data)
            print(data, "is produce.")
    def resume():
        for _ in range(num):
            data = buffer.get()
            print(data, "is resume.")
    th0 = threading.Thread(target=produce, args=(0, ))
    th1 = threading.Thread(target=produce, args=(1, ))
    th2 = threading.Thread(target=resume)
    th3 = threading.Thread(target=resume)
    th0.start()
    th1.start()
    th2.start()
    th3.start()
    th0.join()
    th1.join()
    th2.join()
    th3.join()
    print("The test is End!!!")
'''
'''
# 客户端直接利用文件传输 时间消耗少
import requests

# url = "139.159.179.226:9092"
datas = {"reportEventId": "8487f6faef55904e6b3ec8843a75051c", "robotNumber": "ROB23040224", "typeName": "门窗检测'", "event_pos": "2.5224,-1.5382",
"event_type": "2",
"disVoice": "0",
"file_type": "2",
"floor": "25",
"map": "01",
"posName": "门窗",
"identify": "18",
"robotId": "3157e730186d46b385b4cfb38570ffbd",
"url": "https://file.test.concoai.com/image/3.jpg",
"typeCode": "E0015'",
"disShowFlg": "1",
"imgUrl": "https://file.test.concoai.com/image/3.jpg",
"eventTypeId": "81b5cf4bee494187b3bd82db6663807",
"disMsg": "1",
"disPlay": "1",
"rName": "室内1号",
"tenantId": "2c0ba80be39b46a1a6368c0938deb196",
"priorityCode": "HIGH",
"operationType": "18",
"cmd": "202",
"vehicleid": "ROB23040224",
"time": "20210806150102",
"result": "True",
"points": "[[0,0,100,100],[200,200,300,300]]"}
r = requests.post("https://resttest.concoai.com/inDoorServer/identify/", json=datas)
print(r.content)
print(r.status_code)
'''

'''
import cv2

cap = cv2.VideoCapture("https://file.test.concoai.com/image/000248ec0307d7d09942af00e3840851.jpg")
_, img = cap.read()
cap.release()
cv2.imshow("cap", img)
cv2.waitKey(0)
'''


from kafka import KafkaProducer
import datetime
import json

# 启动生产者
producer = KafkaProducer(bootstrap_servers='139.159.179.226:9092', api_version=(0, 10, 2))
my_topic = "test"

for i in range(10):
    # data = {'num': i, 'data': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    url = 'http://file.test.concoai.com/test_image/1_' + str(i+1) + '.jpeg'
    data = {'reportEventId': '67098c0e9a674d4c86f905d94f43178c',
            'code': '0',
            'robotNumber': 'ROB23040224',
            'typeName': '灭火器检测',
            'event_pos': '-0.8773,-0.0839',
            'event_type': '2',
            'routeId': 'e4ebe777b1a3043a773fbe3519a2850e',
            'pointId': '灭火器',
            'disVoice': '0',
            'file_type': '2',
            'mapName': 'ccc',
            'floor': 29,
            'map': 'ccc',
            'posName': '灭火器',
            'routeNumber': 'MAP23060141',
            'identify': '14',
            'fromUserId': 'ROB23040224',
            'mapNumber': 'MAP23060141',
            'message': 'SUCCESS',
            'robotId': '3157e730186d46b385b4cfb38570ffbd',
            'url': url,
            'typeCode': 'E0013',
            'disShowFlg': '1',
            'imgUrl': url,
            'eventTypeId': '6aef3934e8ac43a6be8f1f2f100512ab',
            'disMsg': '1',
            'disPlay': '1',
            'rName': '室内1号',
            'tenantId': '3da7c466d385448fb6a161d6ab5c8c64',
            'priorityCode': 'HIGH',
            'operationType': '14',
            'cmd': '202',
            'time': 1686386146589,
            'vehicleid': 'ROB23040224'}
    producer.send(my_topic, json.dumps(data).encode('utf-8')).get(timeout=30)


'''
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'test',
    group_id='test_id',
    bootstrap_servers=['139.159.179.226:9092'],  # 要发送的kafka主题
    # auto_offset_reset='earliest',  # 有两个参数值，earliest和latest，如果省略这个参数，那么默认就是latest
)
for msg in consumer:
    print(msg)
'''


'''
from kafka import KafkaConsumer, TopicPartition
import threading
from concurrent.futures import ThreadPoolExecutor

# 创建消费者
consumer = KafkaConsumer(bootstrap_servers=['139.159.179.226:9092'], group_id='test_id')
topic = 'test'
# 获取topic的分区
partition = list(consumer.partitions_for_topic(topic=topic))[0]
print(partition)
topic_partition = TopicPartition(topic=topic, partition=partition)
# 指定消费的topic
consumer.assign([topic_partition])
for msg in consumer:
    print(msg)
'''





