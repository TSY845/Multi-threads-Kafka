import requests
import os
import time
import cv2
import numpy as np
import urllib.request

# 图片链接
# image_url = "https://file.test.concoai.com/image/000248ec0307d7d09942af00e3840851.jpg"
image_url = "https://file.test.concoai.com/image/3.jpg"

# 检查请求是否成功
res = urllib.request.urlopen(image_url)
status_code = res.getcode()
if status_code == 200:
    print("Request successful.")
else:
    raise Exception("Request failed with status code %d" % status_code)

img = np.asarray(bytearray(res.read()), dtype="uint8")
img = cv2.imdecode(img, cv2.IMREAD_COLOR)
cv2.imshow("test", img)
cv2.waitKey(0)

# # 下载图片
# parsing_url = urlparse(image_url)
# img_path = parsing_url.path
# img_name = os.path.basename(img_path)
# dirPath = "D:/Intern/Kafka/recv_img/"
# with open(dirPath+img_name, mode="wb") as f:
#     f.write(r.content)  # 图片内容写入文件
#     print('%s saved to %s' % (img_name, dirPath))
