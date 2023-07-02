import cv2
import numpy as np
import os
import time
from PIL import Image
from torchvision import transforms


def cosine_similarity(image1, image2):
    # 余弦相似度
    transform = transforms.Compose([transforms.Resize((image1.shape[0], image1.shape[1]))])
    image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    image2 = transform(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(np.average(pixel_tuple))
        vectors.append(vector)
        norms.append(np.linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = np.dot(a / a_norm, b / b_norm)
    return res


def pHash(img):
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]
    hash = []
    average = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > average:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def ham_dist(x, y):
    """
    Get the hamming distance of two values.
        hamming distance(汉明距)
    :param x:
    :param y:
    :return: the hamming distance
    """
    assert len(x) == len(y)
    return sum([ch1 != ch2 for ch1, ch2 in zip(x, y)])

def p_hash_similarity(img1, img2):
    """
    Return the pHash score of two images.
    :param img1: image 1
    :param img2: image 2
    :return: (integer) the pHash similarity score
    """
    hash1 = pHash(img1)
    hash2 = pHash(img2)
    pHash_distance = ham_dist(hash1, hash2)
    return pHash_distance


if __name__ == "__main__":
    template_path = "/home/dl/Downloads/Kafka/1/1.jpg"  # 模板路径
    template = cv2.imread(template_path)  # 模板图像
    test_path = "/home/dl/Downloads/Kafka/1/2.jpg"    # 测试路径
    test_img = cv2.imread(test_path)
    threshold = 3  # pHash阈值
    pHash_distance = p_hash_similarity(template, test_img)
    print('Image: {}\t pHash Similarity: {}'.format(test_path, pHash_distance))
    if pHash_distance <= threshold:
        print("The door is closed.")
    else:
        print("The door is open or blocked.")