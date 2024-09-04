""""
@author:lpj
彩色图像的 灰度化、二值化
"""

# 常用视觉库
"""
 skimage即是Scikit-Image。基于python脚本语言开发的数字图片处理包，比如PIL,Pillow, opencv, scikit-image等。
 PIL和Pillow只提供最基础的数字图像处理，功能有限；opencv实际上是一个c++库，只是提供了python接口，更新速度非常慢。
 scikit-image是基于scipy的一款图像处理包，它将图片作为numpy数组进行处理，正好与matlab一样，因此，我们最终选择scikit-image进行数字图像处理。
"""
from skimage.color import rgb2gray

""""
 NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


################################
#  灰度化
################################
img = cv2.imread("D:\cv_workspace\picture\lenna.png")  # 绝对路径 opencv读进来的图片是BGR 需要转成 RGB
'''
 Python 中常见的切片操作，sequence[start:end:step]： 左闭右开 
                            sequence 是要提取元素的序列（例如字符串、列表、元组等）
                            start 是提取的起始位置（包含），默认为 0
                            end 是提取的结束位置（不包含），默认为序列长度
                            step 是提取的步长，默认为 1
    当 start 和 end 都不指定时，[:2] 表示从序列的第一个元素开始，提取两个元素。
'''
h, w = img.shape[:2]  # 获取图片的high和wide   左闭右开

img_gray = np.zeros([h, w], img.dtype)  # 生成和当前图片大小([h,w])、类型(dtype)一样的单通道(zeros,全0矩阵)图片

for i in range(h):
    for j in range(w):
        # 取出high和wide中的BGR坐标     m为彩色图片中 当前像素点的值 [ 125 137 226]
        m = img[i, j]
        # 将BGR坐标转化成gray坐标并赋值给新图像  img_gray[i, j]为灰度图中当前像素点的值 126
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
        '''
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                r, g, b = img[i, j, :]
                # 最大值灰度化
                max_gray[i, j] = max(r, g, b)
                # 平均值灰度化
                ave_gray[i, j] = (r + g + b) / 3
                # 加权平均值灰度化
                weight_gray[i, j] = 0.3 * r + 0.59 * g + 0.11 * b
        '''
print(m)
print(img_gray[i, j])
print("image show gray[i,j]: %s" % img_gray)
cv2.imshow("image show gray", img_gray)
cv2.waitKey(10000)  # 显示10s



####################################
#  原图  画布221
####################################
plt.subplot(221)  # 使用plt.subplot来创建小图,plt.subplot(221)表示将整个图像窗口分为2行2列,当前位置为1
img = plt.imread("D:\cv_workspace\picture\lenna.png")

# img = cv2.imread("lenna.png", False)
"""
plt.imshow()是Matplotlib中的一个函数，用于显示图像。它可以传递一个二维或三维数组作为image参数， 并将图像数据显示为图形，并对图像进行不同的可视化设置。
            cmap：颜色设置。常用的值有’viridis’、‘gray’、'hot’等。可以通过plt.colormaps()查看可用的颜色映射。
            aspect：调整坐标轴。这将根据图像数据自动调整坐标轴的比例。常用的值有’auto’、'equal’等。设置为’auto’时会根据图像数据自动调整纵横比，而设置为’equal’时则会强制保持纵横比相等。
            interpolation：插值方法。它定义了图像在放大或缩小时的插值方式。常用的值有’nearest’、‘bilinear’、'bicubic’等。较高的插值方法可以使图像看起来更平滑，但计算成本更高。
            alpha：透明度。它允许您设置图像的透明度，取值范围为0（完全透明）到1（完全不透明）之间。
            vmin和vmax：用于设置显示的数据值范围。当指定了这两个参数时，imshow()将会根据给定的范围显示图像，超出范围的值会被截断显示。
"""
plt.imshow(img)
print("---image lenna----")
print(img)


########################################
#  灰度化 直接调用Api 将图转化成灰度图 画布222
########################################
'''
rgb2gray是matlab内部一种处理图像的函数，通过消除图像色调和饱和度信息同时保留亮度实现将RGB图像或彩色图转换为灰度图像，即灰度化处理的功能，
         调用这个功能的格式是I = rgb2gray(RGB)，意思是将真彩色图像RGB转换为灰度强度图像I 。
'''
img_gray = rgb2gray(img)
'''
ndarray.shape = (512, 512)   ndarray.dtype = float32
    [[0.60802865 0.60802865 0.60779065 ... 0.6413741  0.57998234 0.46985728]
     [0.60802865 0.60802865 0.60779065 ... 0.6413741  0.57998234 0.46985728]
     [0.60802865 0.60802865 0.60779065 ... 0.6413741  0.57998234 0.46985728]
     ...
'''
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.cvtColor() 函数是 OpenCV 中用于图像颜色空间转换的函数。它允许你将图像从一个色彩空间转换为另一个色彩空间,可以使用这个函数来实现不同色彩空间之间的转换。
#     cv2.cvtColor(src, code[, dst[, dstCn]])
#             src：输入图像，可以是 NumPy 数组或 OpenCV 中的 Mat 对象。
#             code：颜色空间转换代码，表示目标色彩空间。可以使用 OpenCV 中的 cv2.COLOR_* 常量来指定，如 cv2.COLOR_BGR2GRAY 表示将 BGR 彩色图像转换为灰度图像。

# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)
# plt.show()


########################################
#  二值化  画布223
########################################
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
'''
ndarray.shape = (512, 512)   ndarray.dtype = float32
    [[1 1 1 ... 1 1 0]
     [1 1 1 ... 1 1 0]
     [1 1 1 ... 1 1 0]
     ...
'''
print(img_gray.shape, img_gray.dtype)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
