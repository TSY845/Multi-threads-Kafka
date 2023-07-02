from kafka import KafkaConsumer
from concurrent.futures import ThreadPoolExecutor, as_completed
from kafka.structs import TopicPartition
import cv2
import time
import torch
import requests
import threading
import numpy as np
import urllib.request
from check_similarity import p_hash_similarity
from yolov5_62.detect_new import run
from extract_contours import extract_contours, select_rect_upright


def read_img_from_url(url):
    # 从url中读取图片(cv2格式)
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


def http_post(data):
    # 发送HTTP Post
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
    if task == "14":
        # 消防设备(灭火器)检测
        with torch.inference_mode():
            result, points = run(
                weights=
                "D:\\Intern\\Kafka\\yolov5_62\\yolov5s_emergency_equipment.pt",
                source=img_path,
                imgsz=(640, 640),
                conf_thres=0.7,
                device='0',
                classes=0)
        result_dict["result"] = str(result)
        result_dict["points"] = str(points)
    elif task == "15":
        # 门窗检测
        template_path = "D:\\Intern\\datasets\\door\\1\\1.jpg"  # 模板路径
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
        # result_dict["result"] = "True"
        result_dict["points"] = str([])
    elif task == "16":
        # 入侵检测
        with torch.inference_mode():
            result, points = run(
                weights="D:\\Intern\\Kafka\\yolov5_62\\yolov5s.pt",
                source=img_path,
                imgsz=(640, 640),
                conf_thres=0.75,
                device='0',
                classes=0)
        result_dict["result"] = str(result)
        result_dict["points"] = str(points)
    elif task == "17":
        # 安全出口标志检查
        # 检测
        with torch.inference_mode():
            result, points = run(
                weights=
                "D:\\Intern\\Kafka\\yolov5_62\\yolov5s_emergency_equipment.pt",
                source=img_path,
                imgsz=(640, 640),
                conf_thres=0.7,
                device='0',
                classes=1)
        if not result:
            print("No exit sign detected.")
            result_dict["result"] = "False"
            result_dict["points"] = str([])
        else:
            x1, y1, x2, y2 = points[0]  # 检测区域左上右下两点坐标
            img = cv2.imread(img_path)
            # 轮廓提取
            img_roi = img[int(y1):int(y2), int(x1):int(x2)]  # 裁切出检测到标志牌的区域
            contours = extract_contours(img_roi.copy(), draw=False)
            cutouts, coordinates = select_rect_upright(img_roi,
                                                       contours,
                                                       draw=False)  # 获取轮廓提取结果
            if cutouts:
                # 模板匹配
                flag = 0  # 模板匹配是否成功
                for i in range(len(cutouts)):
                    cutout = cutouts[i]
                    x, y, w, h = coordinates[i]  # 轮廓提取后外接矩形相对检测区域的坐标
                    x_min, y_min = int(x1 + x), int(y1 + y)
                    x_max, y_max = int(x_min + w), int(y_min + h)
                    '''在此处加入物体位移条件判断语句'''
                    # img_new = img.copy()
                    # cv2.rectangle(img_new, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
                    # cv2.imshow("whole_cutout", img_new) # 在原图中标识出轮廓提取区域
                    # cv2.waitKey()
                    template_path = "D:\\Intern\\datasets\\exit_sign\\extracted\\template.jpg"  # 模板路径
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
                    print("Template matching failed.")
                    result_dict["result"] = "False"
                    result_dict["points"] = str([])
            else:
                print("Contour extraction failed.")
                result_dict["result"] = "False"
                result_dict["points"] = str([])
    else:
        raise Exception(f"Invalid task number: {task}")
    return result_dict


class MultiThreadKafka(object):
    # 消费者端多线程Kafka
    def __init__(self, max_workers=2):
        try:
            self.consumer = KafkaConsumer(
                bootstrap_servers='139.159.179.226:9092')
            topic = 'test'
            partition = list(
                self.consumer.partitions_for_topic(topic=topic))[0]
            print('partition: ', partition)
            self.tp = TopicPartition(topic=topic,
                                     partition=partition)  # (topic, partition)
            self.consumer.assign([self.tp])
            start_offset = self.consumer.beginning_offsets([self.tp])[self.tp]
            end_offset = self.consumer.end_offsets([self.tp])[self.tp]
            print('earliest offset: ', start_offset)
            print('latest offset: ', end_offset)
            self.seek = end_offset - 10  # 默认获取最新offset
            self.max_workers = max_workers
            # self.consumer.seek_to_end(self.tp)
            # for i in range(start_offset, end_offset):
            #     msg = next(self.consumer)
            #     print(msg)
            #     print(msg.value)
        except Exception as e:
            print(e)

    def operate(self):
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
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        if img is None:
            raise Exception(f"No image read from {url}.")
        tmp_path = "D:/Intern/Kafka/img_temp/temp.jpg"
        cv2.imwrite(tmp_path, img)
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
    # img_path = "D:\\Intern\\Kafka\\test_images\\extinguisher.png"
    # result_dict = {}
    # task = "14"
    # result_dict = assign_tasks(img_path, task, result_dict)
    # print(result_dict)

    # task 15
    # img_path = "D:\\Intern\\datasets\\door\\1\\12.jpg"
    # result_dict = {}
    # task = "15"
    # result_dict = assign_tasks(img_path, task, result_dict)
    # print(result_dict)

    # task 16
    # img_path = "D:\\Intern\\Kafka\\test_images\\human.jpg"
    # result_dict = {}
    # task = "16"
    # result_dict = assign_tasks(img_path, task, result_dict)
    # print(result_dict)

    # task 17
    # img_path = r"D:\Intern\datasets\exit_sign\robot\1.jpeg"
    # result_dict = {}
    # task = "17"
    # result_dict = assign_tasks(img_path, task, result_dict)
    # print(result_dict)
