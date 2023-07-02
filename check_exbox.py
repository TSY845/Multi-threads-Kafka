# -*- encoding: utf-8 -*-
'''
@Description: 灭火器箱正确摆放检测
@Date: 2023/06/08 11:33:15
@Author: Bohan Yang
@version: 1.0
'''
import os
import cv2
import math
import numpy as np
from scipy import ndimage
from ppocr_main import PaddleOCR
from fnmatch import fnmatch


def show_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey()


def to_binary(image, draw=False):
    """
    @Description:
    ---------
    将图像转换为二值图
    @Params: 
    -------
    image: 图像

    draw: 是否绘制中间图像
    @Returns: 
    -------
    binary: 二值图
    """
    # 灰度化
    image_copy = image.copy()
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # 对图像进行中值滤波
    result = cv2.medianBlur(gray, 3)

    # 转化为二值图
    binary = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 10)
    if draw:
        cv2.namedWindow("binary")
        cv2.imshow('binary', binary)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return binary


def rotate(image, angle):
    """
    @Description:
    ---------
    以给定角度旋转图片
    @Params: 
    -------
    image: 输入图像

    angle: 旋转角(逆时针)
    @Returns: 
    -------
    rotated_img: 旋转后图像
    """
    rotated_img = ndimage.rotate(image, angle, mode='nearest')
    return rotated_img


def gray_std_dev(img):
    """
    @Description: 
    ---------
    计算灰度图的均值和方差
    @Params: 
    -------
    img: 原始图像(BGR)
    @Returns: 
    -------
    mean: 灰度均值
    
    std: 灰度方差
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    return mean[0][0], std[0][0]


def extract_contours(image, draw=False):
    """
    @Description: 
    ---------
    从图像中提取所有轮廓
    @Params:
    -------
    images: 图像
    
    draw: 是否绘制中间图像
    @Returns: 
    -------
    contours: 所有轮廓的列表
    """
    image_copy = image.copy()
    binary = to_binary(image)
    # 针对二值图像获取轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)
    if draw:
        cv2.drawContours(image_copy, contours, -1, (0, 0, 255), 2)
        cv2.namedWindow("contours")
        cv2.imshow('contours', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return contours


def select_rect_upright(image, contours, area_thres, aspect_thres, draw=False):
    """
    @Description: 
    ---------
    从全部轮廓中挑选适当轮廓,并截取其正外接矩形
    @Params: 
    -------
    image: 原始图像

    contours: 轮廓

    area_thres: 面积比阈值,接受两元素tuple或list,分别表示面积比下限和上限

    aspect_thres: 纵横比阈值,接受两元素tuple或list,分别表示纵横比下限和上限

    draw: 是否打印中间图像
    @Returns: 
    -------
    cutouts: 原图中截取的图片列表

    coords: 截取的图片在原图中的坐标(xmin, ymin, xmax, ymax)组成的列表
    """
    assert len(
        area_thres
    ) == 2, 'area_thres must be a list or tuple that contains two elements.'
    assert len(
        aspect_thres
    ) == 2, 'aspect_thres must be a list or tuple that contains two elements.'
    total_area = image.shape[0] * image.shape[1]  # 全图面积
    cutouts = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)  # 轮廓面积比
        area_ratio = contour_area / total_area
        if area_ratio < area_thres[0] or area_ratio > area_thres[1]:
            continue
        # 直边界正矩形
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w  # 纵横比
        if aspect_ratio < aspect_thres[0] or aspect_ratio > aspect_thres[1]:
            continue
        cutout = image[y:y + h, x:x + w]
        cutouts.append(cutout)
        if draw:
            cv2.namedWindow("cutout")
            cv2.imshow('cutout', cutout)
            cv2.waitKey(0)
    return cutouts


def get_rotate_angle(binary, draw=False):
    """
    @Description: 
    ---------
    利用Hough变换获取图像中的直线,并以各直线平均倾角作为整幅图的倾角
    @Params: 
    -------
    binary: 二值图

    draw: 是否绘制中间图像
    @Returns: 
    -------
    avg_rotate_angle: 图像倾角
    """
    # Canny算子(阈值可使用canny_threshold函数调整)
    edges = cv2.Canny(binary, 150, 300, apertureSize=3)
    # Hough变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    sum = 0
    count = 0
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            if x2 != x1:
                t = float(y2 - y1) / (x2 - x1)
                if t <= np.pi / 5 and t >= -np.pi / 5:
                    rotate_angle = math.degrees(math.atan(t))
                    sum += rotate_angle
                    count += 1
                    if draw:
                        cv2.line(binary, (x1, y1), (x2, y2), (0, 0, 255),
                                 1)  # 绘制直线

    if count == 0:
        avg_rotate_angle = 0
    else:
        avg_rotate_angle = sum / count
    if draw:
        cv2.imshow("lines", binary)
        cv2.waitKey()
    return avg_rotate_angle


def contour_num(image, contours, area_thres, draw=False):
    """
    @Description: 
    ---------
    统计符合一定面积比的轮廓个数
    @Params: 
    -------
    image: 图像

    contours: 所有轮廓

    area_thres: 面积比阈值,接受两元素tuple或list,分别表示面积比下限和上限

    draw: 是否绘制中间图像
    @Returns: 
    -------
    num: 满足面积比要求的轮廓个数
    """
    assert len(
        area_thres
    ) == 2, 'area_thres must be a list or tuple that contains two elements.'
    total_area = image.shape[0] * image.shape[1]
    image_copy = image.copy()
    num = 0
    for contour in contours:
        image_copy = image.copy()
        contour_area = cv2.contourArea(contour)
        area_ratio = contour_area / total_area
        if area_ratio >= area_thres[0] and area_ratio <= area_thres[1]:
            num += 1
            if draw:
                cv2.drawContours(image_copy, contour, -1, (0, 0, 255), 2)
                cv2.namedWindow("contour")
                cv2.imshow('contour', image_copy)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    return num


def canny_threshold(image):
    """
    @Description: 
    ---------
    可视化调整Canny算子阈值
    @Params: 
    -------
    image: 输入图像
    """
    # 设置窗口
    cv2.namedWindow('Canny')

    # 定义回调函数
    def nothing(x):
        pass

    # 创建两个滑动条，分别控制threshold1，threshold2
    cv2.createTrackbar('threshold1', 'Canny', 50, 400, nothing)
    cv2.createTrackbar('threshold2', 'Canny', 100, 400, nothing)
    while True:
        # 返回滑动条所在位置的值
        threshold1 = cv2.getTrackbarPos('threshold1', 'Canny')
        threshold2 = cv2.getTrackbarPos('threshold2', 'Canny')
        # Canny边缘检测
        img_edges = cv2.Canny(image, threshold1, threshold2)
        # 显示图片
        cv2.imshow('original', image)
        cv2.imshow('Canny', img_edges)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


def check_integrity(image, draw=False):
    """
    @Description: 
    ---------
    主函数。检查图片中灭火器箱是否正确摆放。
    @Params: 
    -------
    image: 图像

    draw: 是否绘制中间图像
    @Returns: 
    -------
    flag: True-灭火器箱正确摆放 False-灭火器箱破损或被遮挡
    """
    flag = False  # True: 灭火器箱正确摆放 False: 灭火器箱破损或被遮挡
    # 第一次提取轮廓
    contours = extract_contours(image)
    cutouts = select_rect_upright(image,
                                  contours,
                                  area_thres=(0.02, 0.9),
                                  aspect_thres=(1.2, 1.6),
                                  draw=draw)
    if len(cutouts) == 0:  # 提取轮廓失败
        flag = False
        return flag
    # 保留一个面积最大的来计算倾角
    elif len(cutouts) == 1:
        largest_cutout = cutouts[0]
    else:
        max_area = 0
        for i in range(len(cutouts)):
            cutout = cutouts[i]
            area = cutout.shape[0] * cutout.shape[1]
            if area > max_area:
                max_area = area
                largest_cutout = cutout
    # show_image(largest_cutout, "largest_cutout")
    # 获取倾角
    rotate_angle = get_rotate_angle(to_binary(largest_cutout), draw=draw)
    if abs(rotate_angle) > 10:  # 倾角过大
        flag = False
        return flag
    # 倾斜校正
    corrected_image = rotate(image, rotate_angle)
    # 对校正后的图像做轮廓提取
    contours = extract_contours(corrected_image, draw=draw)
    cutouts = select_rect_upright(corrected_image,
                                  contours,
                                  area_thres=(0.01, 0.5),
                                  aspect_thres=(1.2, 1.6),
                                  draw=draw)
    if len(cutouts) == 0:  # 提取轮廓失败
        flag = False
        return flag
    # 有可能提取出多个轮廓，只要有一个通过判别即认为指示牌正常
    for cutout in cutouts:
        # OCR文字识别
        ocr_predictor = PaddleOCR(det_model_path="/home/dl/Downloads/Kafka/paddle_ocr_v2/ch_PP-OCRv2_det_infer/inference.pdmodel",
                                  rec_model_path="/home/dl/Downloads/Kafka/paddle_ocr_v2/ch_PP-OCRv2_rec_infer/inference.pdmodel",
                                  character_dict_path="/home/dl/Downloads/Kafka/paddle_ocr_v2/dict/ppocr_keys_v1.txt",
                                  cls_model_path=None,
                                  use_space_char=True)
        boxes_batch, texts = ocr_predictor.predict(cutout)
        if len(texts) != 2:  # 没有提取出两行字
            flag = False
            continue
        for [text_conf] in texts:  # （文本, 置信度)二元组
            keyword_found = False
            text = text_conf[0]
            if fnmatch(text, "火警*119"):
                keyword_found = True
            elif fnmatch(text, "灭火器箱"):
                keyword_found = True
            if not keyword_found:
                flag = False
                continue
        # 对检测到的文字区域涂背景色
        for box in boxes_batch[0]['points']:
            box = box.astype(np.int32)
            left, top = box[0, 0], box[0, 1]
            right, bottom = box[2, 0], box[2, 1]
            cutout = cv2.rectangle(cutout, (left, top), (right, bottom), (38, 29, 96), -1)
        cutout_contours = extract_contours(cutout, draw=draw)
        num = contour_num(cutout, cutout_contours, area_thres=(0.05, 0.5), draw=draw)
        if num == 0:
            flag = True
            return flag
        else:
            flag = False
    return flag


def check_integrity2(roi, draw=False):
    """
    @Description:
    ---------
    主函数。检查图片中灭火器箱是否正确摆放。
    @Params:
    -------
    roi: 根据检测模型结果提取出的指示牌区域图像

    draw: 是否绘制中间图像
    @Returns:
    -------
    flag: True-灭火器箱正确摆放 False-灭火器箱破损或被遮挡
    """
    flag = False  # True: 灭火器箱正确摆放 False: 灭火器箱破损或被遮挡
    # 获取倾角
    rotate_angle = get_rotate_angle(to_binary(roi), draw=draw)
    if abs(rotate_angle) > 10:  # 倾角过大
        flag = False
        return flag
    # 倾斜校正
    corrected_image = rotate(roi, rotate_angle)
    # 对校正后的图像做轮廓提取
    # contours = extract_contours(corrected_image, draw=draw)
    # cutouts = select_rect_upright(corrected_image,
    #                               contours,
    #                               area_thres=(0.5, 0.99),
    #                               aspect_thres=(1.2, 1.6),
    #                               draw=draw)
    # if len(cutouts) == 0:  # 提取轮廓失败
    #     flag = False
    #     return flag
    cutouts = [corrected_image]
    # 有可能提取出多个轮廓，只要有一个通过判别即认为指示牌正常
    for cutout in cutouts:
        # OCR文字识别
        ocr_predictor = PaddleOCR(det_model_path="/home/dl/Downloads/Kafka/paddle_ocr_v2/ch_PP-OCRv2_det_infer/inference.pdmodel",
                                  rec_model_path="/home/dl/Downloads/Kafka/paddle_ocr_v2/ch_PP-OCRv2_rec_infer/inference.pdmodel",
                                  character_dict_path="/home/dl/Downloads/Kafka/paddle_ocr_v2/dict/ppocr_keys_v1.txt",
                                  cls_model_path=None,
                                  use_space_char=True)
        boxes_batch, texts = ocr_predictor.predict(cutout)
        keywords_dict = {"火警": 0, "119": 0, "灭火器箱": 0}
        for [text_conf] in texts:  # （文本, 置信度)二元组
            text = text_conf[0]
            if fnmatch(text, "*火警*"):
                keywords_dict['火警'] += 1
            if fnmatch(text, "*119*"):
                keywords_dict['119'] += 1
            if fnmatch(text, "*灭火器箱*"):
                keywords_dict['灭火器箱'] += 1
        if keywords_dict['火警'] == 1 and keywords_dict['119'] == 1 and keywords_dict['灭火器箱'] == 1:
            flag = True
        else:
            flag = False
        # 对检测到的文字区域涂背景色
        # for box in boxes_batch[0]['points']:
        #     box = box.astype(np.int32)
        #     left, top = box[0, 0], box[0, 1]
        #     right, bottom = box[2, 0], box[2, 1]
        #     cutout = cv2.rectangle(cutout, (left, top), (right, bottom), (38, 29, 96), -1)
        # cutout_contours = extract_contours(cutout, draw=draw)
        # num = contour_num(cutout, cutout_contours, area_thres=(0.2, 0.5), draw=draw)
        # cv2.drawContours(cutout, cutout_contours, -1, (0,0,255), 2)
        # cv2.imwrite("cutout.png", cutout)
        # if num == 0:
        #     flag = True
        #     return flag
        # else:
        #     flag = False
    return flag


if __name__ == '__main__':
    folder_path = r"D:\Intern\datasets\fire_extinguisher_box\0608"
    txt_path = r'D:\Intern\datasets\fire_extinguisher_box'  # 测试结果保存路径
    f = open(f'{txt_path}/test_result.txt', 'w')
    file_list = os.listdir(folder_path)
    for file in file_list:
        print(file)
        image_path = folder_path + "/" + file
        # 读取图片
        image = cv2.imread(image_path)
        flag = check_integrity(image, draw=False)
        f.write("Image: {}\t Integrity: {}\n".format(file, flag))
        print(flag)
    f.close()
