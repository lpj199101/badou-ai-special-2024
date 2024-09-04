import cv2
import numpy as np

'''
  插值算法：求虚拟点的像素值
    已知像素点位置    --(缩放比例)-->  虚拟点像素位置
    已知像素点像素值  ------------->  虚拟点像素值
'''
def function(img):
    height, width, channels = img.shape  # 原图的 height、width、channels(3通道)
    emptyImage = np.zeros((800, 800, channels), np.uint8)  # 放大成 800X800 通道数不变:3通道
    sh = 800 / height  # sh 放大倍数: 800/512
    sw = 800 / width  # sw 放大倍数: 800/512
    for i in range(800):  # i 为放大后图像的坐标
        for j in range(800):  # j 为放大后图像的坐标
            x = int(i / sh + 0.5)  # x 原始图像的坐标    int(),转为整型，使用向下取整
            y = int(j / sw + 0.5)  # y 原始图像的坐标
            """
            例：     2.2    2.5    2.7
              +0.5  2.7    3      3.2   int(放大后坐标/放大比例+0.5）效果等于四舍五入   
                    2      3      3
            """
            emptyImage[i, j] = img[x, y]  # 经过计算后原始图像上的坐标 赋值给 放大图像的坐标
    return emptyImage


# cv2.resize(img, (800,800,c),near/bin)  直接调用api
img = cv2.imread("D:\cv_workspace\picture\lenna.png")  # 读取原图 512X512  shape=(512,512,3)  dtype=unit8
zoom = function(img)  # 定义function函数实现上采样
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)



'''
    `cv2.imread` 是 OpenCV 库中的一个函数，用于从文件加载图像。
    
       入参：cv2.imread(path, flags=cv2.IMREAD_COLOR)
            其中，`path` 是图像文件的路径，
                 `flags` 是一个可选参数，用于指定图像的读取方式：
                        - `cv2.IMREAD_COLOR`：加载彩色图像（默认）
                           1：彩色模式（Color），图像以彩色形式加载，每个像素由红、绿、蓝三个分量组成，每个分量的值在 0 到 255 之间。
                        - `cv2.IMREAD_GRAYSCALE`：加载灰度图像  
                           0：灰度模式（Grayscale），图像以灰度形式加载，每个像素的值在 0 到 255 之间。
                        - `cv2.IMREAD_UNCHANGED`：加载原始图像，不进行任何转换
                           -1：原始模式（Original），图像以原始格式加载，不进行任何转换。
                           
       反参：cv2.imread 函数返回的是一个 `numpy.ndarray` 对象 img, 以多维数组的形式保存图片信息，其中包含了图像的像素数据。
            【img[i,j]像素值】 可以通过img[i,j]获取图像该像素点的像素值：你可以通过索引 img 数组来访问图像的像素值。例如，img[i, j] 可以获取位于坐标 (i, j) 的像素值。
            【img.shape】可以通过函数返回的`numpy.ndarray`对象的`shape`属性来获取图像的格式和大小，
                           返回的值为 `(height, width, channels)`，其中前两维表示图片的像素坐标 `height` 和 `width` 分别表示图像的高度和宽度，
                           最后一维表示图片的通道索引，具体图像的通道数由图片的格式来决定`channels`表示图像的通道数（通常为 1 或 3，分别表示灰度图像和彩色图像）。
            【img.dtype】可以通过函数返回的`numpy.ndarray`对象的`dtype`属性来获取图像数据的类型和精度，返回的 img.dtype 通常是 numpy.uint8 或类似的无符号整数类型。
                         `cv2.imread`函数通常会返回以下数据类型之一： 
                            - 8 位整数（`np.uint8`）：表示每个像素的值范围为 0 到 255，这是最常见的图像数据类型之一，适用于大多数图像格式。 √
                            - 16 位整数（`np.uint16`）：表示每个像素的值范围为 0 到 65535，这种类型通常用于一些特殊的图像格式或需要更高精度的应用。
                            - 32 位整数（`np.uint32`）：表示每个像素的值范围为 0 到 4294967295，这种类型也比较常见，尤其是在一些高清图像或需要更高精度的应用中。
                            - 浮点数（`np.float32`或`np.float64`）：表示每个像素的值为浮点数，这种类型通常用于一些科学计算或需要更高精度的应用。
                    np.uint8(RGB24格式):
                        RGB三个色彩通道，每个通道有8位的数据，等级(灰阶)是0～255共256级(2^8)，即色精度为8位，最后通过rgb三个通道加色原理表示，所以一共是24位.
                        BGR色彩空间中第1个8位（第1个字节）存储的是蓝色组成信息（Blue component），第2个8位（第 2 个字节）存储的是绿色组成信息（Green component），
                        第3个8位（第3个字节）存储的是红色组成信息（Red component）。同样，其第 4 个、第 5 个、第 6 个字节分别存储蓝色、绿色、红色组成信息，以此类推。
'''



'''
np.zeros是NumPy库中的一个函数，用于创建一个指定形状（shape）和数据类型（dtype）的全零数组。
numpy.zeros(shape, dtype=float, order='C')
     shape: 这是一个必需参数，指定了数组的维度。例如，shape=3 创建一个长度为 3 的一维数组，而 shape=(2,3) 则创建一个 2x3 的二维数组。
     dtype: 这个参数允许用户指定数组中元素的数据类型。常见类型包括 numpy.int32, numpy.float64 等。如果不指定，NumPy 默认使用 float64 类型。img.dtype='unit8'
     order: 不常用，允许高级用户优化他们的数组布局，以便更高效地进行特定的数学运算。可选参数，可以是’C’（按行排列）或’F’（按列排列）。大多数情况下，默认的 'C' 顺序就足够了。
'''


'''
在 Python 中，`range()` 函数用于生成一个整数序列：
    range(stop)
    range(start, stop, step)
        - `start`：表示序列的起始位置（包含该位置），默认为 0。
        - `stop`：表示序列的结束位置（不包含该位置）。
        - `step`：表示序列的步长，默认为 1。
    
    例如： range(5) 会生成一个包含 0 到 4 的整数序列：0 1 2 3 4
    例如： range(1, 11, 2) 会生成一个奇数序列 1 3 5 7 9 
    
使用 `list()` 函数将 `range()` 函数生成的序列转换为列表：  list(range(5)) ->  [0, 1, 2, 3, 4]
'''




'''
`cv2.imshow()` 是 OpenCV 库中的一个函数，用于在指定的窗口中显示图像：

cv2.imshow(winname, mat)

    其中，`winname` 是窗口的名称，`mat` 是要显示的图像：
        1. **窗口名称**：`winname` 是窗口的名称，用于标识显示图像的窗口。你可以根据需要为窗口指定一个名称。
        2. **图像数据**：`mat` 是要显示的图像。它可以是一个 NumPy 数组，表示灰度图像或彩色图像。对于彩色图像，通常使用 BGR 颜色顺序。
        
        3. **窗口显示**：当调用 `cv2.imshow()` 函数时，它会在指定的窗口中显示图像。你可以调整窗口的大小、位置和其他属性，以便更好地查看图像。
        4. **键盘交互**：在显示图像的窗口中，你可以通过按下键盘上的按键来进行交互。例如，按下 `Esc` 键可以关闭窗口。
需要注意的是，`cv2.imshow()` 函数是阻塞式的，它会一直等待用户的操作，直到用户关闭显示图像的窗口。在实际应用中，你可能需要在显示图像的同时进行其他处理，这时可以使用多线程或异步编程的方式来实现。
'''