import os
import sys
import random
import math
import numpy as np
import skimage.io
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from nets.mrcnn import get_predict_model
from utils.config import Config
from utils.anchors import get_anchors
from utils.utils import mold_inputs, unmold_detections
from utils import visualize
import keras.backend as K


class MASK_RCNN(object):

    # ------------------------------------------------------------------------------------------#
    #   0.将类的默认属性字典 self._defaults 更新到对象的属性字典 self.__dict__ 中,这样 对象就拥有了默认的属性
    # ------------------------------------------------------------------------------------------#
    _defaults = {
        "model_path": 'model_data/mask_rcnn_coco.h5',  # 模型文件的路径
        "classes_path": 'model_data/coco_classes.txt',  # 类别文件的路径
        "confidence": 0.7,  # 置信度阈值

        # 使用coco数据集检测的时候，IMAGE_MIN_DIM=1024，IMAGE_MAX_DIM=1024, RPN_ANCHOR_SCALES=(32, 64, 128, 256, 512)
        "RPN_ANCHOR_SCALES": (32, 64, 128, 256, 512),  # 一个包含不同锚点尺度的元组
        "IMAGE_MIN_DIM": 1024,  # 图像的最小维度
        "IMAGE_MAX_DIM": 1024,  # 图像的最大维度

        # 在使用自己的数据集进行训练的时候，如果显存不足要调小图片大小
        # 同时要调小anchors
        # "IMAGE_MIN_DIM": 512,
        # "IMAGE_MAX_DIM": 512,
        # "RPN_ANCHOR_SCALES": (16, 32, 64, 128, 256)
    }

    '''
    定义了一个类方法 `get_defaults`，用于获取指定属性的默认值:    
        1. `@classmethod` 装饰器：这个装饰器将下面的方法标记为类方法，可以通过类本身而不是类的实例来调用。
        2. `get_defaults(cls, n)` 方法：这个方法接受两个参数，`cls` 表示类本身，`n` 表示要获取默认值的属性名称。
        3. 条件判断：在方法内部，通过检查属性名称 `n` 是否在类的默认属性字典 `cls._defaults` 中，来确定是否存在对应的默认值。
        4. 返回结果：如果属性名称存在于默认属性字典中，就返回对应的默认值；否则，返回一个包含错误信息的字符串，提示属性名称未被识别。
    通过这个类方法，你可以方便地获取类的默认属性值，而不需要直接访问类的内部属性字典。如果属性名称不存在，还可以得到一个有意义的错误提示。
    '''
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ----------------------------------------------------------------------------------------------------------------------------------#
    #   【初始化Mask-Rcnn】
    # ----------------------------------------------------------------------------------------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # 0.将类的默认属性字典 self._defaults 更新到对象的属性字典 self.__dict__ 中,这样 对象就拥有了默认的属性
        self.class_names = self._get_class()  # 1.获得所有的分类：调用 _get_class 方法获取类的名称，并将其存储在 self.class_names 属性中
        self.sess = K.get_session()  # 获取当前的 TensorFlow 会话，并将其存储在 self.sess 属性中
        self.config = self._get_config()  # 2.调用 _get_config 方法获取配置信息，并将其存储在 self.config 属性中
        self.generate()  # 3. 生成模型

    # -------------------------------------------------------------------------------#
    #   1.获得所有的分类：调用 _get_class 方法获取类的名称，并将其存储在 self.class_names 属性中
    # -------------------------------------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)  # 获取类文件的路径，并使用 `os.path.expanduser` 函数将其扩展为完整的用户目录路径
        with open(classes_path) as f:  # 打开类文件，并将文件对象赋值给变量 `f`  classes_path:'model_data/coco_classes.txt'
            # 读取类文件中的所有行，并将它们存储在列表 `class_names` 中  {list:80}['person\n', 'bicycle\n', 'car\n', ... , 'toothbrush\n']
            class_names = f.readlines()
        # .strip() 方法用于移除字符串两端的空格或指定的字符
        # 对列表中的每个元素（类名称）进行处理，使用 `strip` 方法去除前后的空格和换行符 {list:80}['person', 'bicycle', 'car', ..., 'toothbrush']
        class_names = [c.strip() for c in class_names]
        class_names.insert(0, "BG")  # 在类名称列表的开头插入一个名为 "BG" 的元素 {list:81}['BG', 'person', 'bicycle', ..., 'toothbrush']
        return class_names  # 返回处理后的类别名称列表

    # -------------------------------------------------------------------------------#
    #   2.调用 _get_config 方法获取配置信息，并将其存储在 self.config 属性中
    # -------------------------------------------------------------------------------#
    def _get_config(self):  # 定义了一个`_get_config` 的方法，用于获取配置信息
        class InferenceConfig(Config):  # 定义了一个`InferenceConfig` 的类，它继承自 `Config` 类
            NUM_CLASSES = len(self.class_names)  # 类的数量，通过计算 `self.class_names` 的长度来确定
            GPU_COUNT = 1  # GPU 的数量，设置为 1
            IMAGES_PER_GPU = 1  # ：每个 GPU 处理的图像数量，设置为 1
            DETECTION_MIN_CONFIDENCE = self.confidence  # 检测的最小置信度，通过 `self.confidence` 获取  0.7
            NAME = "shapes"  # 名称，设置为 "shapes"
            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES  # 锚点的尺度 {tuple:5} (32, 64, 128, 256, 512)
            IMAGE_MIN_DIM = self.IMAGE_MIN_DIM  # 图像的最小维度  1024
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM  # 图像的最大维度  1024

        config = InferenceConfig()  # 创建了一个 `InferenceConfig` 对象，并将其赋值给变量 `config`
        config.display()  # 显示配置信息
        return config  # 返回配置对象 `config`

    # ---------------------------------------------------#
    #   3.生成模型
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)  # 'model_data/mask_rcnn_coco.h5'
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算总的种类
        self.num_classes = len(self.class_names)  # 81

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入
        # 否则先构建模型再载入
        self.model = get_predict_model(self.config)
        self.model.load_weights(self.model_path, by_name=True)

    # ---------------------------------------------------#
    #   4.检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):  # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1330x1330 at 0x19CB7C21208>
        # 将输入的图像转换为 NumPy 数组，并将其封装在一个列表中
        image = [np.array(image)]  #
        # 使用 mold_inputs 函数对图像进行预处理，得到处理后的图像、图像元数据和窗口信息
        '''
        [array([[[255, 255, 255],
        ...,
        [ 41,  50,  57],
        [ 43,  54,  60],
        [ 43,  54,  60]],
        ...
        [108, 108, 110]]], dtype=uint8)]   [[[[131.3 138.2 151.1],   [131.3 138.2 151.1],   [131.3 138.2 151.1],   ...,   [-83.7 -67.8 -47.9],   [-81.7 -64.8 -44.9],   [-80.7 -62.8 -43.9]],,  [[131.3 138.2 151.1],   [131.3 138.2 151.1],   [131.3 138.2 151.1],   ...,   [-79.7 -63.8 -43.9],   [-79.7 
        '''
        molded_images, image_metas, windows = mold_inputs(self.config, image)
        # 获取处理后图像的形状  (1024,1024,3)
        image_shape = molded_images[0].shape
        # 根据图像形状获取锚点
        '''
        使用 NumPy 的 `broadcast_to` 函数将 `anchors` 数组广播到新的形状。
            `np.broadcast_to` 函数的作用是将输入数组按照指定的形状进行广播，使其在维度上与目标形状匹配。在广播过程中，数组的元素会根据需要进行复制，以填充目标形状中的每个位置。
            在这段代码中，`anchors` 是一个数组，`(1,) + anchors.shape` 是一个新的形状，其中 `1` 表示在第一个维度上添加一个长度为 1 的维度。
            通过调用 `np.broadcast_to(anchors, (1,) + anchors.shape)`，将 `anchors` 数组广播到新的形状，从而在第一个维度上添加了一个长度为 1 的维度。
            目的是为了在后续的计算中对 `anchors` 数组进行扩展或添加额外的维度，以便与其他数组进行操作或计算。            
        '''
        anchors = get_anchors(self.config, image_shape)
        # 将锚点广播到与处理后图像相同的形状。
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)
        # 使用模型对处理后的图像进行预测，得到检测结果、掩码等信息
        '''
        从 `self.model.predict` 函数中获取返回值的一部分，并将它们分别赋值给变量 `detections`、`_`、`_`、`mrcnn_mask`、`_`、`_` 和 `_`:
            - `detections`：这是模型的检测结果，可能包含关于检测到的对象的信息，如位置、类别、置信度等。
            - `mrcnn_mask`：这是模型生成的掩码（mask），用于表示检测到的对象的形状或轮廓。
            - `_`（下划线）：这些下划线表示未使用或不重要的返回值。在这种情况下，可能是模型的其他输出或中间结果，但在当前代码中没有被进一步处理或使用。
        总的来说，这段代码从模型的预测结果中提取了检测结果和掩码，并将它们分别存储在 `detections` 和 `mrcnn_mask` 变量中，而其他返回值则被忽略。这些结果可以用于后续的处理、分析或可视化，具体取决于应用的需求。
        \ ： 表示续行符，反斜杠表示将下一行的代码视为当前行的延续 ，如果需要将一行代码拆分成多行，可以使用反斜杠作为续行符
        '''
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.model.predict([molded_images, image_metas, anchors], verbose=0)

        # 使用 unmold_detections 函数将预测结果转换为最终的检测框、类别 ID、得分和掩码
        final_rois, final_class_ids, final_scores, final_masks = \
            unmold_detections(detections[0], mrcnn_mask[0],
                              image[0].shape, molded_images[0].shape,
                              windows[0])
        # 将检测结果存储在一个字典中
        r = {
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }

        # 画图(开源工具) 使用可视化工具显示检测结果，包括图像、检测框、掩码、类别 ID 和得分
        visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])

    def close_session(self):
        self.sess.close()
