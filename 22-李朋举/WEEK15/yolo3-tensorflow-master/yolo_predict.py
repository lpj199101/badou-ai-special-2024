import os
import config
import random
import colorsys
import numpy as np
import tensorflow as tf
from model.yolo3_model import yolo


class yolo_predictor:

    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        """
        Introduction
        ------------
            初始化函数
        Parameters
        ----------
            obj_threshold: 目标检测为物体的阈值
            nms_threshold: nms阈值
        """
        self.obj_threshold = obj_threshold  # 0.5
        self.nms_threshold = nms_threshold  # 0.5
        # 预读取
        self.classes_path = classes_file  # './model_data/coco_classes.txt'
        self.anchors_path = anchors_file  # './model_data/yolo_anchors.txt'
        # 读取种类名称
        self.class_names = self._get_class()  # {list:80} ['person', 'bicycle', 'car', 'motorbike',  ... , 'toothbrush']
        # 读取先验框
        self.anchors = self._get_anchors()  # (9,2) [[ 10.  13.], [ 16.  30.],..., [373. 326.]]

        # 画框用
        # 创建一个包含色相值（Hue）的元组列表 hsv_tuples。其中，色相值是通过将类别数量除以元组的索引值再乘以一个比例因子得到的。
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]

        # 使用 map() 函数将 hsv_tuples 中的每个元组传递给 colorsys.hsv_to_rgb() 函数，将色相值转换为 RGB 颜色值。然后，将转换后的颜色值列表赋值给实例变量 self.colors。
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # 使用 map() 函数将 self.colors 中的每个颜色值转换为整数，并将结果赋值给实例变量 self.colors
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        '''
        `random.seed(10101)` 是为了在程序中设置随机数生成器的种子
            在 Python 中，`random` 模块用于生成随机数。当使用 `random` 模块生成随机数时，每次运行程序时生成的随机数序列可能会有所不同，这是因为随机数生成器的初始状态是不确定的。
            通过设置随机数生成器的种子，可以确保在每次运行程序时生成的随机数序列都是相同的。这在需要重复实验或生成可重现的随机结果时非常有用。
            
            在上述代码中，`random.seed(10101)` 将随机数生成器的种子设置为 10101。这意味着在后续使用 `random` 模块生成随机数时，将基于这个种子生成相同的随机数序列。
            例如，如果你在代码中多次调用 `random.randint(1, 10)`，每次调用都会返回相同的随机整数，因为随机数生成器的种子是相同的。
            注意，设置随机数生成器的种子只会影响后续使用 `random` 模块生成的随机数，而不会影响其他代码中的随机行为。如果你在程序的其他部分使用了不同的随机数生成器或设置了不同的种子，它们将产生不同的随机结果。
        '''
        # 设置随机数生成器的种子为 10101，以确保每次运行代码时生成的随机数序列是相同的
        random.seed(10101)
        # 使用 random.shuffle() 函数打乱 self.colors 列表中元素的顺序，以实现随机颜色的效果
        random.shuffle(self.colors)
        # 将随机数生成器的种子设置为 None，以便在后续的代码中使用系统默认的随机数生成器。
        random.seed(None)

    def _get_class(self):
        """
        Introduction
        ------------
            读取类别名称
        """
        # 获取类别名称文件的路径。os.path.expanduser 用于将路径中的 ~ 或 $HOME 等用户目录缩写扩展为完整的用户目录路径
        classes_path = os.path.expanduser(self.classes_path)
        # 以读模式打开类别名称文件，并将文件对象赋值给变量 f。
        with open(classes_path) as f:
            # 通过调用文件对象的 readlines 方法，将文件中的每一行内容作为一个元素存储在列表 class_names 中
            class_names = f.readlines()
        # 使用列表推导式对 class_names 列表中的每个元素进行处理。strip 方法用于去除元素两端的空格和换行符。
        class_names = [c.strip() for c in class_names]
        # 函数返回处理后的类别名称列表
        return class_names

    def _get_anchors(self):
        """
        Introduction
        ------------
            读取anchors数据
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            # 读取文件中的第一行内容，并将其赋值给变量 anchors
            anchors = f.readline()
            # 使用列表推导式将 anchors 字符串按照逗号分割，并将每个分割后的部分转换为浮点数，存储在列表 anchors 中
            anchors = [float(x) for x in anchors.split(',')]
            # 将列表 anchors 转换为 numpy 数组，并使用 reshape 方法将其重塑为一个二维数组，其中每行包含两个 anchor 的值。
            # 将anchors数组重塑为一个二维数组，其中第二维的大小为 2。第一维的大小由-1表示，这意味着 numpy 会根据anchors数组的元素数量和第二维的大小来自动计算第一维的大小。
            #          例如，如果anchors数组有 6 个元素，那么reshape(-1, 2)将把它重塑为一个 3x2 的二维数组
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    # ---------------------------------------#
    #   对三个特征层解码
    #   进行排序并进行非极大抑制
    # ---------------------------------------#
    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        Introduction
        ------------
            将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        Parameters
        ----------
            feats: yolo输出的feature map
            anchors: anchor的位置
            class_num: 类别数目
            input_shape: 输入大小
            image_shape: 图片大小
        Returns
        -------
            boxes: 物体框的位置
            boxes_scores: 物体框的分数，为置信度和类别概率的乘积
        """
        # 获得特征
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        # 寻找在原图上的位置
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        # 获得类别置信度 box_confidence * box_class_probs    标定框的置信度 X 当前目标属于每个框的分类概率
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])  # Tensor("predict/Reshape_5:0", shape=(?, 80), dtype=float32)
        return boxes, box_scores

    # 获得在原图上框的位置
    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        Introduction
        ------------
            计算物体框预测坐标在原图中的位置坐标
        Parameters
        ----------
            box_xy: 物体框左上角坐标
            box_wh: 物体框的宽高
            input_shape: 输入的大小
            image_shape: 图片的大小
        Returns
        -------
            boxes: 物体框的位置
        """

        '''
        将 box_xy 和 box_wh 的最后一个维度的顺序反转，这是因为在坐标表示中，通常是以 (y, x) 的顺序表示的，而在 TensorFlow 中，维度的顺序是 (x, y)，所以需要进行反转
        '''
        # 将 box_xy 的最后一个维度的顺序反转 Tensor("predict/strided_slice_15:0", shape=(?, ?, ?, 3, 2), dtype=float32)
        box_yx = box_xy[..., ::-1]
        # 将 box_wh 的最后一个维度的顺序反转 Tensor("predict/strided_slice_16:0", shape=(?, ?, ?, 3, 2), dtype=float32)
        box_hw = box_wh[..., ::-1]

        # 416,416
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        # 实际图片的大小, 将输入形状和图片形状转换为 float32 类型  Tensor("predict/Cast_4:0", shape=(2,), dtype=float32)
        image_shape = tf.cast(image_shape, dtype=tf.float32)

        # 计算新的形状，通过将图片形状乘以输入形状与图片形状的最小值的比例，并使用 tf.round 函数进行四舍五入。
        # Tensor("predict/Round:0", shape=(2,), dtype=float32)
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

        # 计算偏移量和比例，偏移量用于将坐标从输入形状的坐标系转换到新形状的坐标系，比例用于将坐标从新形状的坐标系转换到原图的坐标系。
        # Tensor("predict/truediv_4:0", shape=(2,), dtype=float32)
        offset = (input_shape - new_shape) / 2. / input_shape
        # Tensor("predict/truediv_5:0", shape=(2,), dtype=float32)
        scale = input_shape / new_shape

        # 将 box_yx 减去偏移量，然后乘以比例，将 box_hw 乘以比例，这是为了将坐标从输入形状的坐标系转换到原图的坐标系。
        # Tensor("predict/mul_3:0", shape=(?, ?, ?, 3, 2), dtype=float32)
        box_yx = (box_yx - offset) * scale
        # Tensor("predict/mul_4:0", shape=(?, ?, ?, 3, 2), dtype=float32)
        box_hw *= scale

        # 计算框的最小坐标和最大坐标，通过将 box_yx 减去 box_hw 的一半和加上 box_hw 的一半。
        box_mins = box_yx - (box_hw / 2.)  # Tensor("predict/sub_2:0", shape=(?, ?, ?, 3, 2), dtype=float32)
        box_maxes = box_yx + (box_hw / 2.)  # Tensor("predict/add_2:0", shape=(?, ?, ?, 3, 2), dtype=float32)

        # 将框的最小坐标和最大坐标沿着最后一个维度进行连接，得到一个形状为 (..., 4) 的张量。
        boxes = tf.concat([  # Tensor("predict/concat_1:0", shape=(?, ?, ?, 3, 4), dtype=float32)
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)

        # 将 boxes 乘以图片形状，这是为了将坐标从原图的坐标系转换到实际的像素坐标系。 Tensor("predict/mul_5:0", shape=(?, ?, ?, 3, 4), dtype=float32)
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    # 其实是解码的过程
    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        Introduction
        ------------
            根据yolo最后一层的输出确定bounding box
        Parameters
        ----------
            feats: yolo模型最后一层输出        特征1： Tensor("predict/yolo/conv2d_59/BiasAdd:0", shape=(?, 13, 13, 255), dtype=float32)
            anchors: anchors的位置(质心)      [[116.  90.], [156. 198.], [373. 326.]]  (yolo_anchors中索引为6,7,8的数据)
            num_classes: 类别数量             80
            input_shape: 输入大小             Tensor("predict/mul:0", shape=(2,), dtype=int32)
        Returns
        -------
            box_xy, box_wh, box_confidence, box_class_probs
        """
        num_anchors = len(anchors)  # 3
        # Tensor("predict/Reshape:0", shape=(1, 1, 1, 3, 2), dtype=float32)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        # 获取张量feats的第 2 维和第3 维的尺寸，并将其存储在变量grid_size中  Tensor("predict/strided_slice_1:0", shape=(2,), dtype=int32)
        grid_size = tf.shape(feats)[1:3]  # 13,13
        # 对张量 feats 进行重整形 Tensor("predict/Reshape_1:0", shape=(?, ?, ?, 3, 85), dtype=float32)
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标 Tensor("predict/Tile:0", shape=(?, ?, 1, 1), dtype=int32)
        '''
        tf.range() 是 TensorFlow 中的一个函数，用于创建一个数值序列张量, 函数接受一个或多个参数,用于指定序列的起始值、结束值和步长:
           例：tf.range(0, 10, 2) -> 这将创建一个从 0 到 8 的偶数序列张量，步长为 2  
           
        1. `tf.range(grid_size[0])` ：这创建了一个从 0 到 `grid_size[0] - 1` 的整数序列张量。
        2. `tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1])` ：将序列张量重塑为形状为 `[-1, 1, 1, 1]` 的张量。`-1` 表示该维度的大小将根据其他维度的大小自动计算。
        3. `tf.tile(..., [1, grid_size[1], 1, 1])` ：对重塑后的张量进行复制，使其在第二个维度上重复 `grid_size[1]` 次。
        4. 最终的结果 `grid_y` 是一个形状为 `[grid_size[0], grid_size[1], 1, 1]` 的张量，其中包含了从 0 到 `grid_size[0] - 1` 的整数序列，在第二个维度上重复了 `grid_size[1]` 次。
        总结起来，这段代码的作用是创建一个包含垂直方向上的网格坐标的张量 `grid_y`。这个张量可以用于在图像处理或其他任务中表示网格的 y 坐标。  
        '''
        # Tensor("predict/Tile:0", shape=(?, ?, 1, 1), dtype=int32)
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        # Tensor("predict/Tile_1:0", shape=(?, ?, 1, 1), dtype=int32)
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        # Tensor("predict/concat:0", shape=(?, ?, 1, 2), dtype=int32)
        grid = tf.concat([grid_x, grid_y], axis=-1)
        # 将张量的数据类型转换为指定的数据类型 Tensor("predict/Cast:0", shape=(?, ?, 1, 2), dtype=float32)
        grid = tf.cast(grid, tf.float32)

        # 将x,y坐标归一化，相对网格的位置
        '''
        进行坐标的计算：
            1. `predictions[..., :2]`：这是对 `predictions` 张量进行切片操作，选取最后两个维度的所有元素。通常，这表示预测的边界框的中心坐标（`x` 和 `y`）。
            2. `tf.sigmoid(predictions[..., :2])`：对中心坐标进行 `sigmoid` 激活函数处理。`sigmoid` 函数将输入值压缩到 `0` 到 `1` 的范围内，这有助于将预测的坐标限制在有效的范围内。
            3. `grid`：这是之前定义的网格张量，它包含了网格的坐标信息。 
            4. `tf.sigmoid(predictions[..., :2]) + grid`：将处理后的中心坐标与网格张量相加。这实际上是将预测的坐标与网格中的位置进行对齐。
            5. `tf.cast(grid_size[::-1], tf.float32)`：将 `grid_size` 张量进行类型转换为 `tf.float32`，并将其顺序反转。这是为了与前面的加法操作保持一致。
            6. `(tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)`：将对齐后的坐标除以网格大小，得到相对坐标。这将坐标归一化到 `0` 到 `1` 的范围内，表示边界框在网格中的相对位置。    
        将预测的边界框中心坐标转换为相对坐标，以便在后续的处理中使用。通过 `sigmoid` 激活函数和与网格的对齐操作，实现了坐标的归一化和限制在有效范围内。
        '''
        # Tensor("predict/truediv:0", shape=(?, ?, ?, 3, 2), dtype=float32)
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)

        # 将w,h也归一化
        '''
        预测的边界框宽度和高度的计算：
            1. `predictions[..., 2:4]`：这是对 `predictions` 张量进行切片操作，选取最后两个维度的从第 2 个元素到第 4 个元素的子张量。通常，这表示预测的边界框的宽度和高度。
            2. `tf.exp(predictions[..., 2:4])`：对宽度和高度进行指数运算。`tf.exp` 函数将输入值取指数，这有助于将预测的宽度和高度转换为正数。
                   tf.exp(x, name=None), x 是一个张量，表示要计算指数的输入值,  函数返回一个与输入张量 x 具有相同形状的张量，其中每个元素都是 e 的 x 次幂
            3. `anchors_tensor`：这是之前定义的锚框张量，它包含了锚框的宽度和高度信息。 shape=(1, 1, 1, 3, 2)
            4. `tf.exp(predictions[..., 2:4]) * anchors_tensor`：将指数运算后的宽度和高度与锚框张量相乘。这实际上是将预测的宽度和高度与锚框进行比较和调整。
            5. `tf.cast(input_shape[::-1], tf.float32)`：将输入形状张量 `input_shape` 进行类型转换为 `tf.float32`，并将其顺序反转。这是为了与前面的乘法操作保持一致。
                    `input_shape[::-1]` 是 Python 中的切片操作，用于反转一个张量的形状。在这个例子中，`input_shape` 是一个张量，表示输入图像的形状。`[::-1]` 表示从最后一个元素开始，以步长为-1 的方式进行切片，即反转整个张量的形状。
                    例如，如果 `input_shape` 是 `(32, 32, 3)`，那么 `input_shape[::-1]` 就是 `(3, 32, 32)`。                    
            6. `tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)`：将调整后的宽度和高度除以输入形状的倒数，得到相对宽度和高度。这将宽度和高度归一化到相对于输入图像的大小。
        将预测的边界框宽度和高度转换为相对大小，以便在后续的处理中使用。通过指数运算、与锚框的比较和归一化操作，实现了对边界框大小的调整和归一化。
        '''
        # Tensor("predict/truediv_1:0", shape=(?, ?, ?, 3, 2), dtype=float32)
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)

        # 计算置信度
        '''
        使用 TensorFlow 的 `tf.sigmoid` 函数对 `predictions` 张量的最后一个维度的第 4 个元素到第 5 个元素进行了 `sigmoid` 激活, 
        sigmoid` 函数将输入值压缩到 0 到 1 之间，通常用于将模型的输出转换为概率或置信度：
            `box_confidence` 张量将包含预测的边界框的置信度值，这些值经过了 `sigmoid` 激活，使得它们在 0 到 1 之间。
            这样的处理常用于目标检测任务中，其中模型需要对每个边界框预测一个置信度，表示该边界框中包含目标的可能性。
            通过将置信度值限制在 0 到 1 之间，可以更直观地解释和处理这些值，例如设置阈值来确定哪些边界框被认为是有效的检测结果。            
        '''
        # Tensor("predict/Sigmoid_1:0", shape=(?, ?, ?, 3, 1), dtype=float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        '''
        使用 TensorFlow 的 `tf.sigmoid` 函数对 `predictions` 张量的最后一个维度的第 5 个元素到最后一个元素进行了 `sigmoid` 激活。
        '''
        # Tensor("predict/Sigmoid_2:0", shape=(?, ?, ?, 3, 80), dtype=float32)
        box_class_probs = tf.sigmoid(predictions[..., 5:])

        return box_xy, box_wh, box_confidence, box_class_probs

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        """
        Introduction
        ------------
            根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
        Parameters
        ----------
            yolo_outputs: yolo模型输出  {list:3:  Tensor("predict/yolo/conv2d_59/BiasAdd:0", shape=(?, 13, 13, 255), dtype=float32)  Tensor("predict/yolo/conv2d_67/BiasAdd:0", shape=(?, 26, 26, 255), dtype=float32)  Tensor("predict/yolo/conv2d_75/BiasAdd:0", shape=(?, 52, 52, 255), dtype=float32)
            image_shape: 图片的大小     Tensor("Placeholder:0", shape=(2,), dtype=int32)
            max_boxes:  最大box数量     20
        Returns
        -------
            boxes_: 物体框的位置
            scores_: 物体类别的概率
            classes_: 物体类别
        """
        # 每一个特征层对应三个先验框  (3个特征层公9个先验框)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # 存储每个特征层的预测框和对应的分数
        boxes = []
        box_scores = []
        # inputshape是416x416
        # image_shape是实际图片的大小(输入图像的大小，通过 YOLO 模型的输出形状计算得到)   Tensor("predict/mul:0", shape=(2,), dtype=int32)
        input_shape = tf.shape(yolo_outputs[0])[1: 3] * 32

        # 对三个特征层的输出获取每个预测box坐标和box的分数，score = 置信度x类别概率
        # ---------------------------------------#
        #   对三个特征层解码
        #   获得分数和框的位置
        # ---------------------------------------#
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]], len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)  # _boxes -> Tensor("predict/Reshape_4:0", shape=(?, 4), dtype=float32)
            box_scores.append(_box_scores)  # _box_scores -> Tensor("predict/Reshape_5:0", shape=(?, 80), dtype=float32)
        # 将所有特征层的预测框和分数连接在一起, 放在一行里面便于操作
        boxes = tf.concat(boxes, axis=0)  # Tensor("predict/concat_9:0", shape=(?, 4), dtype=float32)
        box_scores = tf.concat(box_scores, axis=0)  # Tensor("predict/concat_10:0", shape=(?, 80), dtype=float32)

        mask = box_scores >= self.obj_threshold  # obj_threshold 0.5
        # 创建了一个常量张量 max_boxes_tensor，其值为 max_boxes，数据类型为 tf.int32
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []

        # ---------------------------------------#
        #   1、取出每一类得分大于self.obj_threshold
        #   的框和得分
        #   2、对得分进行非极大抑制
        # ---------------------------------------#
        # 对每一个类别进行判断(80个, coco数据集)
        for c in range(len(self.class_names)):
            # 取出所有类为c的box  (使用布尔掩码从 boxes 中选择出所有类别为 c 的框)
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有类为c的分数  (使用布尔掩码从 box_scores 中选择出所有类别为 c 的分数)
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            # 非极大抑制 (对类别为 c 的框和分数进行非极大值抑制)
            #  max_boxes_tensor 用于限制最大框的数量iou_threshold 是交并比（Intersection over Union，IoU）的阈值，用于判断两个框是否重叠。
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=self.nms_threshold)

            # 获取非极大抑制的结果

            class_boxes = tf.gather(class_boxes, nms_index)  # 使用 tf.gather 函数根据非极大值抑制的结果选择出最终保留的框
            class_box_scores = tf.gather(class_box_scores, nms_index)  # 使用 tf.gather 函数根据非极大值抑制的结果选择出最终保留的分数
            classes = tf.ones_like(class_box_scores, 'int32') * c  # 创建一个与保留的分数长度相同的类别张量，其中所有元素都设置为类别

            boxes_.append(class_boxes)  # 将当前类别的框添加到 boxes_ 列表中
            scores_.append(class_box_scores)  # 将当前类别的分数添加到 scores_ 列表中
            classes_.append(classes)  # 将当前类别的类别添加到 classes_ 列表中

        boxes_ = tf.concat(boxes_, axis=0)  # 将 boxes_ 列表中的框连接成一个张量
        scores_ = tf.concat(scores_, axis=0)  # 将 scores_ 列表中的分数连接成一个张量
        classes_ = tf.concat(classes_, axis=0)  # 将 classes_ 列表中的类别连接成一个张量
        return boxes_, scores_, classes_  # 返回连接后的框、分数和类别的张量

    # ---------------------------------------#
    #   predict用于预测，分三步
    #   1、建立yolo对象
    #   2、获得预测结果
    #   3、对预测结果进行处理
    # ---------------------------------------#
    def predict(self, inputs, image_shape):
        """
        Introduction
        ------------
            构建预测模型
        Parameters
        ----------
            inputs: 处理之后的输入图片   Tensor("Placeholder_1:0", shape=(?, 416, 416, 3), dtype=float32)
            image_shape: 图像原始大小   Tensor("Placeholder:0", shape=(2,), dtype=int32)
        Returns
        -------
            boxes: 物体框坐标
            scores: 物体概率值
            classes: 物体类别
        """
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train=False)
        # yolo_inference用于获得网络的预测结果
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training=False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)
        return boxes, scores, classes
