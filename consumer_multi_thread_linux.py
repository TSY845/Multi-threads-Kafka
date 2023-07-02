from kafka import KafkaConsumer
from concurrent.futures import ThreadPoolExecutor, as_completed
from kafka.structs import TopicPartition
import cv2
import time
import json
import socket
import torch
import requests
import threading
import numpy as np
import urllib.request
from check_similarity_new import p_hash_similarity
from yolov5_62.detect_new import run
from extract_contours import extract_contours, select_rectangle


def read_img_from_url(url):
    '''
    从url中读取图片(cv2格式)
    :param url: 图片链接
    :return: img
    '''
    while True:
        res = urllib.request.urlopen(url)
        status_code = res.getcode()
        if status_code == 200:
            print("Request successful.")
            break
        else:
            print("Request failed with status code %d" % status_code)
            print("Retry in 5 seconds...")
            time.sleep(5)  # 连接失败后间隔5s尝试重连

    img = np.asarray(bytearray(res.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img


def socket_transmit(data):
    '''
    向服务器返回数据(Socket)
    :param data
    :return:
    '''
    HOST = 'localhost'  # or 127.0.0.1
    PORT = 2856
    BUFSIZ = 1024
    retry = 10  # 重试次数
    interval = 5  # 重试间隔(s)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
        for i in range(retry):
            try:
                c.connect((HOST, PORT))
                c.send(data.encode())
                recv_data = c.recv(BUFSIZ)
                print(
                    f'Transmission completed.\nReceived data: {recv_data.decode()}'
                )
                break
            except TimeoutError as e:
                print(f'Socket error: {e}')
            except InterruptedError as e:
                print(f'Socket error: {e}')
            time.sleep(interval)
    if recv_data is None:
        raise TimeoutError('Transmission timed out.')
    return


def http_post(data):
    '''
    发送HTTP Post
    :param data: 待发送数据
    :return: None
    '''
    # url = "https://resttest.concoai.com//docking/identifyEventImage"
    url = "https://resttest.concoai.com/inDoorServer/identify/"
    # headers = {"token": "test"}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()

    except Exception as e:
        print(e)
    status_code = response.status_code
    print('Status Code: {0}'.format(status_code))
    print('Response Text: {0}'.format(response.text))


def assign_tasks(img_path, task, result_dict):
    '''
    为接收到的图片分配并执行多种任务.
    :param img_path: 待测图片路径
    :param task: 任务编号
    :param result_dict: 结果字典
    :return: (dict) result_dict
    '''
    if task == "14":
        # 消防设备(灭火器)检测
        with torch.inference_mode():
            result, points = run(weights="/home/dl/Downloads/Kafka/yolov5_62/yolov5s_emergency_equipment.pt",
                                 source=img_path,
                                 imgsz=(640, 640),
                                 conf_thres=0.7,
                                 device='0',
                                 classes=0
                                 )
        result_dict["result"] = str(result)
        result_dict["points"] = str(points)
    elif task == "15":
        # 门窗检测
        template_path = "/home/dl/Downloads/Kafka/1/1.jpg"  # 模板路径
        template = cv2.imread(template_path)  # 模板图像
        test_img = cv2.imread(img_path)
        threshold = 3  # pHash阈值
        pHash_distance = p_hash_similarity(template, test_img)
        # print('Image: {}\t pHash Similarity: {}'.format(tmp_path, pHash_distance))
        if pHash_distance <= threshold:
            result = "True"
        else:
            result = "False"
        result_dict["result"] = result
        result_dict["points"] = str([])
    elif task == "16":
        # 入侵检测
        with torch.inference_mode():
            result, points = run(weights="/home/dl/Downloads/Kafka/yolov5_62/yolov5s.pt",
                                 source=img_path,
                                 imgsz=(640, 640),
                                 conf_thres=0.75,
                                 device='0',
                                 classes=0
                                 )
        result_dict["result"] = str(result)
        result_dict["points"] = str(points)
    elif task == "17":
        # 安全出口标志检查
        with torch.inference_mode():
            result, points = run(weights="/home/dl/Downloads/Kafka/yolov5_62/yolov5s_emergency_equipment.pt",
                                 source=img_path,
                                 imgsz=(640, 640),
                                 conf_thres=0.7,
                                 device='0',
                                 classes=1
                                 )
        if not result:
            print("No exit sign detected.")  # 未检测到标志牌
            result_dict["result"] = "False"
            result_dict["points"] = str([])
        else:
            x1, y1, x2, y2 = points[0]
            img = cv2.imread(img_path)
            img_roi = img[int(y1):int(y2), int(x1):int(x2)]  # 裁切出检测到标志牌的区域
            contours = extract_contours(img_roi.copy(), draw=False)
            cutouts = select_rectangle(img_roi, contours, draw=False)  # 轮廓提取
            if cutouts:
                flag = 0  # 模板匹配是否成功
                for cutout in cutouts:
                    template_path = "/home/dl/Downloads/Kafka/test_img/exit_sign_template.jpg"  # 模板路径
                    template = cv2.imread(template_path)  # 模板图像
                    test_img = cutout
                    threshold = 10  # pHash阈值
                    pHash_distance = p_hash_similarity(template, test_img)
                    if pHash_distance <= threshold:
                        flag = 1
                if flag:
                    result_dict["result"] = "True"
                    result_dict["points"] = str([])
                else:
                    print("Template matching failed.")  # 模板匹配失败
                    result_dict["result"] = "False"
                    result_dict["points"] = str([])
            else:
                print("Contour extraction failed.")  # 轮廓提取失败
                result_dict["result"] = "False"
                result_dict["points"] = str([])
    else:
        raise Exception("Invalid task!")
    return result_dict


class MultiThreadKafka(object):
    # 消费者端多线程Kafka
    def __init__(self, max_workers=2):
        self.consumer = KafkaConsumer(bootstrap_servers='139.159.179.226:9092')
        topic = 'test'
        partition = list(self.consumer.partitions_for_topic(topic=topic))[0]
        print('partition: ', partition)
        self.tp = TopicPartition(topic=topic, partition=partition)  # (topic, partition)
        self.consumer.assign([self.tp])
        start_offset = self.consumer.beginning_offsets([self.tp])[self.tp]
        end_offset = self.consumer.end_offsets([self.tp])[self.tp]
        print('beginning offset: ', start_offset)
        print('end offset: ', end_offset)
        self.seek = end_offset  # 默认获取最新offset
        self.max_workers = max_workers
        self.consumer.seek_to_end(self.tp)
        # for i in range(start_offset, end_offset):
        #     msg = next(self.consumer)
        #     print(msg)
        #     print(msg.value)

    def operate(self):
        '''
        每条线程执行的操作
        :return: None
        '''
        with lock:
            self.consumer.seek(self.tp, self.seek)
            self.seek += 1
            msg = next(self.consumer)
        msg_value: dict = eval(msg.value.decode("utf-8"))  # 这里把核心数据提取出来
        url = msg_value["url"]  # 图片url
        task = msg_value["identify"]  # 任务编号
        thread_name = threading.current_thread().name  # 线程名
        print("Thread: %s\n [%s:%d:%d]: key=%s value=%s" %
              (thread_name, msg.topic, msg.partition, msg.offset, msg.key,
               msg.value))
        img = read_img_from_url(url)
        if not img:
            raise Exception(f"No image read from {url}.")
        tmp_path = "/home/dl/Downloads/Kafka/img_temp/temp.jpg"
        cv2.imwrite(tmp_path, img)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        # 分任务处理
        result_dict = msg_value
        result_dict = assign_tasks(tmp_path, task, result_dict)
        # print(result_dict)
        http_post(result_dict)
        return

    def main(self):
        thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        while True:  # 无限循环，读取数据
            # thread_mission_list = []
            for i in range(self.max_workers):
                thread = thread_pool.submit(self.operate)
                # thread_mission_list.append(thread)
            # for mission in as_completed(
            #         thread_mission_list):  # 这里会等待线程执行完毕，先完成的会先显示出来
            #     yield mission.result()


if __name__ == '__main__':
    thread_kafka = MultiThreadKafka(max_workers=2)
    lock = threading.Lock()
    kafka_data_generator = thread_kafka.main()  # 迭代器
    # for result in kafka_data_generator:
    #     print(result)

    # task 14
    # img_path = "/home/dl/Downloads/Kafka/test_img/extinguisher.png"
    # result_dict = {}
    # task = "14"
    # result_dict = assign_tasks(img_path, task, result_dict)
    # print(result_dict)

    # task 15
    # img_path = "/home/dl/Downloads/Kafka/1/12.jpg"
    # result_dict = {}
    # task = "15"
    # result_dict = assign_tasks(img_path, task, result_dict)
    # print(result_dict)

    # task 16
    # img_path = "/home/dl/Downloads/Kafka/test_img/human.jpg"
    # result_dict = {}
    # task = "16"
    # result_dict = assign_tasks(img_path, task, result_dict)
    # print(result_dict)

    # task 17
    # img_path = "/home/dl/Downloads/Kafka/test_img/door_sign.png"
    # result_dict = {}
    # task = "17"
    # result_dict = assign_tasks(img_path, task, result_dict)
    # print(result_dict)

