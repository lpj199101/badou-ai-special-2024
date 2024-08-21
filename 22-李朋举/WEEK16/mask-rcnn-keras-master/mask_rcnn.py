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
        with open(classes_path) as f:  # 打开类文件，并将文件对象赋值给变量 `f`
            class_names = f.readlines()  # 读取类文件中的所有行，并将它们存储在列表 `class_names` 中
        class_names = [c.strip() for c in class_names]  # 对列表中的每个元素（类名称）进行处理，使用 `strip` 方法去除前后的空格和换行符
        class_names.insert(0, "BG")  # 在类名称列表的开头插入一个名为 "BG" 的元素
        return class_names  # 返回处理后的类别名称列表

    # -------------------------------------------------------------------------------#
    #   2.调用 _get_config 方法获取配置信息，并将其存储在 self.config 属性中
    # -------------------------------------------------------------------------------#
    def _get_config(self):  # 定义了一个`_get_config` 的方法，用于获取配置信息
        class InferenceConfig(Config):  # 定义了一个`InferenceConfig` 的类，它继承自 `Config` 类
            NUM_CLASSES = len(self.class_names)  # 类的数量，通过计算 `self.class_names` 的长度来确定
            GPU_COUNT = 1  # GPU 的数量，设置为 1
            IMAGES_PER_GPU = 1  # ：每个 GPU 处理的图像数量，设置为 1
            DETECTION_MIN_CONFIDENCE = self.confidence  # 检测的最小置信度，通过 `self.confidence` 获取
            NAME = "shapes"  # 名称，设置为 "shapes"
            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES  # 锚点的尺度，通过 `self.RPN_ANCHOR_SCALES` 获取
            IMAGE_MIN_DIM = self.IMAGE_MIN_DIM  # 图像的最小维度，通过 `self.IMAGE_MIN_DIM` 获取
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM  # 图像的最大维度，通过 `self.IMAGE_MAX_DIM` 获取

        config = InferenceConfig()  # 创建了一个 `InferenceConfig` 对象，并将其赋值给变量 `config`
        config.display()  # 显示配置信息
        return config  # 返回配置对象 `config`

    # ---------------------------------------------------#
    #   3.生成模型
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算总的种类
        self.num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.model = get_predict_model(self.config)
        self.model.load_weights(self.model_path, by_name=True)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image = [np.array(image)]
        molded_images, image_metas, windows = mold_inputs(self.config, image)

        image_shape = molded_images[0].shape
        anchors = get_anchors(self.config, image_shape)
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)

        detections, _, _, mrcnn_mask, _, _, _ = \
            self.model.predict([molded_images, image_metas, anchors], verbose=0)

        final_rois, final_class_ids, final_scores, final_masks = \
            unmold_detections(detections[0], mrcnn_mask[0],
                              image[0].shape, molded_images[0].shape,
                              windows[0])

        r = {
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }

        # 画图(开源工具)
        visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'],
                                    self.class_names, r['scores'])

    def close_session(self):
        self.sess.close()
