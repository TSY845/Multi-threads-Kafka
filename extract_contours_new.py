import os
import cv2


def extract_contours(image, draw=False):
    '''
    从图像中提取轮廓
    @image: 输入图像
    @draw: 是否打印中间图像
    '''
    # 步骤1：图像二值化
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    # 针对二值图像获取轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)
    if draw:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        cv2.namedWindow("contours")
        cv2.imshow('contours', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return contours


def select_rect_upright(image, contours, draw=False):
    '''
    从全部轮廓中挑选适当轮廓的正外接矩形,返回原图对应区域的列表
    @image: 原始图像
    @contours: 轮廓
    @draw: 是否打印中间图像
    '''
    total_area = image.shape[0] * image.shape[1]  # 全图面积
    cutouts, coordinates = [], []
    for contour in contours:
        # 直边界正矩形
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_ratio = area / total_area  # 面积比
        if area_ratio < 0.01 or area_ratio > 0.95:
            continue
        aspect_ratio = h / w  # 纵横比
        if aspect_ratio > 1 or aspect_ratio < 0.2:
            continue
        cutout = image[y:y + h, x:x + w]
        cutouts.append(cutout)
        coordinates.append([x, y, w, h])
        if draw:
            cv2.namedWindow("cutout")
            cv2.imshow('cutout', cutout)
            cv2.waitKey(0)
    return cutouts, coordinates


# def select_rect_rotated(image, contours, draw=False):
#     '''
#     从全部轮廓中挑选适当轮廓的最小旋转外接矩形,返回原图对应区域的列表
#     @image: 原始图像
#     @contours: 轮廓
#     @draw: 是否打印中间图像
#     '''
#     total_area = image.shape[0] * image.shape[1]  # 全图面积
#     result = []
#     for contour in contours:
#         # 直边界正矩形
#         rect = cv2.minAreaRect(contour)
#         x, y, w, h, theta = rect[0], rect[0][1], rect[1][0], rect[1][1], rect[2]
#         area = w * h
#         area_ratio = area / total_area  # 面积比
#         if area_ratio < 0.5 or area_ratio > 0.95:
#             continue
#         aspect_ratio = h / w  # 纵横比
#         if aspect_ratio > 1 or aspect_ratio < 0.2:
#             continue
#         cutout = image[y:y + h, x:x + w] # 这里还没改
#         result.append(cutout)
#         if draw:
#             cv2.namedWindow("cutout")
#             cv2.imshow('cutout', cutout)
#             cv2.waitKey(0)
#     return result


if __name__ == "__main__":
    folder_path = r"D:\Intern\datasets\exit_sign\whole"
    save_path = r"D:\Intern\datasets\exit_sign\robot"
    file_list = os.listdir(folder_path)
    for file in file_list:
        image_path = folder_path + "/" + file
        image = cv2.imread(image_path)
        dirname, filename = os.path.split(image_path)
        filename, extension = os.path.splitext(filename)
        contours = extract_contours(image.copy(), draw=True)
        cutouts, coordinates = select_rect_upright(image, contours, draw=True)  # 正边界矩形筛选
        if not cutouts:
            print(f"{filename}.{extension} 没有适当尺寸的轮廓！")
            continue
        for i in range(len(cutouts)):
            cv2.imwrite(save_path + '\\' + filename + f"_{i}" + extension,
                        cutouts[i])
