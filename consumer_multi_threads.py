# -*- encoding: utf-8 -*-
'''
@Description: Kafka多线程消费端
@Date: 2023/06/12 18:00:00
@Author: Bohan Yang
@version: 2.1
'''

from kafka import KafkaConsumer
from concurrent.futures import ThreadPoolExecutor, as_completed
from kafka.structs import TopicPartition
import cv2
import time
import socket
import torch
import requests
import threading
import numpy as np
import urllib.request
from check_similarity import p_hash_similarity
from yolov5_62.detect_new import run
import check_exit_sign
import check_exbox
import json
import os


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


def assign_tasks(url, task, result_dict):
    '''
    为接收到的图片分配并执行多种任务.
    :param url: 待测图片链接
    :param task: 任务编号
    :param result_dict: 结果字典
    :return: (dict) result_dict
    '''
    img_name = url.split('/')[-1]
    img_save_path = '/home/dl/Downloads/Kafka/img_temp/' + img_name  # 2023.06.10 mxy
    lock = threading.Lock()
    t1 = time.time()
    if task == "14":
        # 2023.06.10 mxy
        # test_img = read_img_from_url(url)
        # cv2.imwrite(img_save_path, test_img)
        ####################################
        # 消防设备(灭火器)检测
        with torch.inference_mode():
            result, points, conf_list = run(
                weights="/home/dl/Downloads/Kafka/yolov5_62/yolov5s_extinguisher_0612_openvino_model/yolov5s_extinguisher_0612.xml",
                source=url,
                imgsz=(640, 640),
                conf_thres=0.8,
                device='cpu',
                nosave=True,
                classes=0)
        result_dict["result"] = str(result)
        result_dict["points"] = str(points)
        file = url.split('/')[-1]
        file_path = os.path.join(os.getcwd(), file)
        # print(file_path)
        if os.path.exists(file_path):
            with lock:
                os.remove(file_path)  # 检测后将缓存的图片删除
    elif task == "15":
        # 门窗检测
        template_path = "/home/dl/Downloads/Kafka/template/temp1.jpg"  # 模板路径
        template = cv2.imread(template_path)  # 模板图像
        test_img = read_img_from_url(url)
        if test_img is None:
            raise Exception(f"No image read from {url}.")
        print("Image received.")
        threshold = 3  # pHash阈值
        pHash_distance = p_hash_similarity(template, test_img)
        # print('Image: {}\t pHash Similarity: {}'.format(tmp_path, pHash_distance))
        if pHash_distance <= threshold:
            result = "True"
        else:
            result = "False"
        result_dict["result"] = result
        result_dict["points"] = str([])
        cv2.imwrite(img_save_path, test_img)
    elif task == "16":
        # 灭火器箱检查
        image = read_img_from_url(url)
        if image is None:
            raise Exception(f"No image read from {url}.")
        print("Image received.")
        # cv2.imwrite(img_save_path, image)
        # result = check_exbox.check_integrity(image)  # 06.14 ybh 老版本灭火器箱检查代码
        ############################
        with torch.inference_mode():
            result, points, conf_list = run(
                weights="/home/dl/Downloads/Kafka/yolov5_62/0615_box_openvino_model/0615_box.xml",
                source=url,
                imgsz=(640, 640),
                conf_thres=0.7,
                device='cpu',
                nosave=True,
                classes=0
                )
        if result:  # 检出灭火器箱
            result_list = []
            for point in points:
                roi = image[int(point[1]):int(point[3]), int(point[0]):int(point[2])]
                flag = check_exbox.check_integrity2(roi)
                result_list.append(str(flag))
            result_dict["result"] = str(result_list)
            result_dict["points"] = str(points)
        else:
            result_list = ["False"] * len(points)
            result_dict["result"] = str(result_list)
            result_dict["points"] = str([])
        file = url.split('/')[-1]
        file_path = os.path.join(os.getcwd(), file)
        # print(file_path)
        if os.path.exists(file_path):
            with lock:
                os.remove(file_path)  # 检测后将缓存的图片删除
        ############################
    elif task == "17":
        # 安全出口标志检查
        image = read_img_from_url(url)
        # if image is None:
        #     raise Exception(f"No image read from {url}.")
        # print("Image received.")
        # cv2.imwrite(img_save_path, image)
        # result = check_exit_sign.check_integrity(image)  # 06.13 ybh 老版本指示牌检查代码
        ############################
        with torch.inference_mode():
            result, points, conf_list = run(
                weights="/home/dl/Downloads/Kafka/yolov5_62/0616_exit_openvino_model/0616_exit.xml",
                source=url,
                imgsz=(640, 640),
                conf_thres=0.7,
                device='cpu',
                nosave=True,
                classes=0
                )
        if result:  # 检出指示牌
            result_list = []
            for point in points:
                roi = image[int(point[1]):int(point[3]), int(point[0]):int(point[2])]
                flag = check_exit_sign.check_integrity3(image, roi, area_thres=(0.05, 0.1), aspect_thres=(0.2, 0.5))
                result_list.append(str(flag))
            result_dict["result"] = str(result_list)
            result_dict["points"] = str(points)
        else:
            result_list = ["False"] * len(points)
            result_dict["result"] = str(result_list)
            result_dict["points"] = str([])
        file = url.split('/')[-1]
        file_path = os.path.join(os.getcwd(), file)
        # print(file_path)
        if os.path.exists(file_path):
            with lock:
                os.remove(file_path)  # 检测后将缓存的图片删除
        ############################
    t2 = time.time()
    print(f"Task {task} running time: {t2 - t1}s.")
    print(result_dict)
    return result_dict


class MultiThreadsKafka(object):
    # 消费者端多线程Kafka
    def __init__(self, max_workers=2):
        self.consumer = KafkaConsumer(bootstrap_servers='139.159.179.226:9092')
        # self.consumer = KafkaConsumer(bootstrap_servers=['139.159.179.226:9092'], group_id='test_id')
        topic = 'test'
        partition = list(self.consumer.partitions_for_topic(topic=topic))[0]
        print('partition: ', partition)
        self.tp = TopicPartition(topic=topic, partition=partition)  # (topic, partition)
        self.consumer.assign([self.tp])
        self.max_workers = max_workers
        start_offset = self.consumer.beginning_offsets([self.tp])[self.tp]
        end_offset = self.consumer.end_offsets([self.tp])[self.tp]
        print('earliest offset: ', start_offset)
        print('latest offset: ', end_offset)
        self.seek = end_offset  # 默认获取最新offset
        self.consumer.seek(self.tp, self.seek)
        #####################################################################
        '''
        # self.consumer.seek_to_beginning(self.tp)
        for i in range(100):
            msg = next(self.consumer)
            print(msg)
            print(msg.value)
        '''

    def operate(self):
        '''
        每条线程执行的操作
        :return: None
        '''
        lock = threading.Lock()
        with open('/home/dl/Downloads/Kafka/1.txt', 'a', encoding='utf-8') as f:  # 2023.06.10 mxy
            with lock:
                # self.consumer.seek(self.tp, self.seek - 1)   # 2023.06.08 mxy  # 2023.06.12 ybh
                # self.seek += 1
                msg = next(self.consumer)
            msg_value: dict = eval(msg.value.decode("utf-8"))  # 这里把核心数据提取出来
            url = msg_value["url"]  # 图片url
            task = msg_value["identify"]  # 任务编号
            thread_name = threading.current_thread().name  # 线程名
            print("Thread: %s\n [%s:%d:%d]: key=%s value=%s" %
                  (thread_name, msg.topic, msg.partition, msg.offset, msg.key,
                   msg.value))

            # 分任务处理
            result_dict = msg_value
            print(f"Assigning task {task}...")
            result_dict = assign_tasks(url, task, result_dict)
            f.write(json.dumps(result_dict, ensure_ascii=False))  # 2023.06.10 mxy
            f.write('\n')
            # print(result_dict)
        f.close()  # 2023.06.10 mxy
        # print("Sending http post...")
        # http_post(result_dict)
        return

    def main(self):
        # thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        while True:  # 无限循环，读取数据
            thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
            # thread_mission_list = []
            for i in range(self.max_workers):
                thread = thread_pool.submit(self.operate)
            thread_pool.shutdown(wait=True)  # 2023.06.09 mxy
            # thread_mission_list.append(thread)
            # for mission in as_completed(
            #         thread_mission_list):  # 这里会等待线程执行完毕，先完成的会先显示出来
            #     yield mission.result()


if __name__ == '__main__':
    # Main Process
    thread_kafka = MultiThreadsKafka(max_workers=12)
    thread_kafka.main()

    # kafka_data_generator = thread_kafka.main()  # 迭代器
    # for result in kafka_data_generator:
    # print(result)

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
    # img_path = "/home/dl/Downloads/Kafka/0608/0608 (6).jpeg"
    # img_path = "/home/dl/Downloads/Kafka/0614_exbox/1.png"
    # url = "http://file.test.concoai.com/image/438bc4d80a8c81982b39174f86d822b9.jpeg"
    # url = "http://file.test.concoai.com/image/d877bae604e988e2cf0b8821fe238c92.jpeg"
    # url = "http://file.test.concoai.com/image/22bb422bff7876c7ff17a522a37249c5.jpeg"
    # url = "http://file.test.concoai.com/image/1f43846bfb90323b789198f287c9f293.jpeg"
    # result_dict = {}
    # task = "16"
    # result_dict = assign_tasks(url, task, result_dict)
    # print(result_dict)


    # task 17
    # # img_path = "/home/dl/Downloads/Kafka/exit_sign_robot/1.jpeg"
    # url = "http://file.test.concoai.com/image/7891820a518684d8cfd39a4a58438a43.jpeg"
    # # url = "http://file.test.concoai.com/image/e06006089d87799d7e921094766307a0.jpeg
    # url = 'http://file.test.concoai.com/image/0ee9ba5f768eeca719acc10884a7cb67.jpeg'
    # result_dict = {}
    # task = "17"
    # result_dict = assign_tasks(url, task, result_dict)
    # print(result_dict)
