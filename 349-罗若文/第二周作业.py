
from skimage.color import rgb2gray    #导入rgb图转灰度图的包
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread("/Users/mac/Desktop/kk.jpg")     #opencv读取图片
print('原始图片的尺寸：')
print(img.shape)   #打印图片的长宽通道
h, w = img.shape[:2]  # 获取图片的high和wide
#shape[0] 为高 ，shape[1]为宽，shape[3]为通道的数量
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
# print("看一下创建的灰度图")
# print(img_gray)
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标,就是img是RGB图片，所以一个像素点有三层，三个值
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像

print(m)  #最后一个像素点的值
print(img_gray)  #打印灰度图各个像素点的值
#print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)  #展示图片，第一个参数是给图片起的名字

plt.subplot(221)  #将画纸切割成两行两列（四个） 在第一个作图
img = plt.imread("/Users/mac/Desktop/kk.jpg")   #matplotlib读取图片
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

# 二值化
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1

img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
