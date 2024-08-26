from keras.layers import Input, ZeroPadding2D, Conv2D, MaxPooling2D, BatchNormalization, Activation, UpSampling2D, Add, \
    Lambda, Concatenate
from keras.layers import Reshape, TimeDistributed, Dense, Conv2DTranspose
from keras.models import Model
import keras.backend as K
from nets.resnet import get_resnet
from nets.layers import ProposalLayer, PyramidROIAlign, DetectionLayer, DetectionTargetLayer
from nets.mrcnn_training import *
from utils.anchors import get_anchors
from utils.utils import norm_boxes_graph, parse_image_meta_graph
import tensorflow as tf
import numpy as np


# ------------------------------------#
#   五个不同大小的特征层会传入到
#   RPN当中，获得建议框
# ------------------------------------#
# feature_map：输入的特征图 Tensor("input_rpn_feature_map:0", shape=(?, ?, ?, 256), dtype=float32)
# anchors_per_location：每个位置的锚点数量 3
def rpn_graph(feature_map, anchors_per_location):
    # Tensor("rpn_conv_shared/Relu:0", shape=(?, ?, ?, 512), dtype=float32)
    shared = Conv2D(512, (3, 3), padding='same', activation='relu', name='rpn_conv_shared')(feature_map)
    # Tensor("rpn_class_raw/BiasAdd:0", shape=(?, ?, ?, 6), dtype=float32)
    x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')(shared)

    # 中间结果 代表这个先验框对应的类  batch_size,num_anchors,2   Tensor("reshape_1/Reshape:0", shape=(?, ?, 2), dtype=float32)
    rpn_class_logits = Reshape([-1, 2])(x)  # 将卷积结果进行重整形，使其变为 [batch_size, num_anchors, 2] 的形状，其中 num_anchors 是锚点的数量

    # RPN分支1  分类结果-对每个分类正负 上面一条通过softmax分类anchors，获得positive和negative分类
    #          Tensor("rpn_class_xxx/truediv:0", shape=(?, ?, 2), dtype=float32)
    rpn_probs = Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)  # 对类别预测结果进行 Softmax 激活，得到每个锚点属于不同类别的概率

    # RPN分支2  回归结果-偏移量  下面一条用于计算对于计算对于 anchors的bouding box regression偏移量，以获取精确的proposal
    #     这个先验框的调整参数  batch_size,num_anchors,4    Tensor("rpn_bbox_pred/BiasAdd:0", shape=(?, ?, ?, 12), dtype=float32)
    x = Conv2D(anchors_per_location * 4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')(shared)
    # 将卷积结果进行重整形，使其变为 [batch_size, num_anchors, 4] 的形状 其中 num_anchors 是锚点的数量
    # Tensor("reshape_2/Reshape:0", shape=(?, ?, 4), dtype=float32)
    rpn_bbox = Reshape([-1, 4])(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


# ------------------------------------#
#   建立建议框网络模型
#   RPN模型
# ------------------------------------#
def build_rpn_model(anchors_per_location, depth):  # anchors_per_location：3  depth：256
    input_feature_map = Input(shape=[None, None, depth], name="input_rpn_feature_map")  # 输入的特征图，其形状为[None, None, depth]
    outputs = rpn_graph(input_feature_map, anchors_per_location)  # input_feature_map-模型的输入, anchors_per_location-每个位置的锚点数量 3
    return Model([input_feature_map], outputs, name="rpn_model")


# -----------------------------------------------#
#   分类： 建立classifier模型
#   回归: 这个模型的预测结果会调整建议框, 获得最终的预测框
# -----------------------------------------------#
def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024):
    """
        rois: Tensor("ROI/packed_2:0", shape=(1, ?, 4), dtype=float32)
        feature_maps: [P2, P3, P4, P5]
        image_meta: Tensor("input_image_meta:0", shape=(?, 93), dtype=float32)  1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES(81)
        pool_size:  7
        num_classes: 81
    """

    # ROI Pooling，利用建议框在特征层上进行截取
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)

    # Shape: [batch, num_rois, 1, 1, fc_layers_size]，相当于两次全连接
    x = TimeDistributed(Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"), name="mrcnn_class_conv1")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = Activation('relu')(x)
    '''
    TimeDistributed:
    对FPN网络输出的多层卷积特征进行共享参数, TimeDistributed的意义在于使不同层的特征图共享权重。
    '''
    # Shape: [batch, num_rois, 1, 1, fc_layers_size]
    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, fc_layers_size]
    shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)

    # Classifier head
    # 这个的预测结果代表这个先验框内部的物体的种类
    mrcnn_class_logits = TimeDistributed(Dense(num_classes),
                                         name='mrcnn_class_logits')(shared)
    mrcnn_probs = TimeDistributed(Activation("softmax"),
                                  name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # 这个的预测结果会对先验框进行调整
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = TimeDistributed(Dense(num_classes * 4, activation='linear'),
                        name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    mrcnn_bbox = Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


# 结果3 mask : ROI Align -> Conv(4个,ppt中2个) -> 上采样到原图(反卷积Conv2DTranspose)
def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    # ROI Align，利用建议框在特征层上进行截取
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                        name="mrcnn_mask_conv1")(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn1')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                        name="mrcnn_mask_conv2")(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                        name="mrcnn_mask_conv3")(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn3')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                        name="mrcnn_mask_conv4")(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn4')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, 2xMASK_POOL_SIZE, 2xMASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                        name="mrcnn_mask_deconv")(x)
    # 反卷积后再次进行一个1x1卷积调整通道，使其最终数量为numclasses，代表分的类
    x = TimeDistributed(Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                        name="mrcnn_mask")(x)
    return x


def get_predict_model(config):
    h, w = config.IMAGE_SHAPE[:2]  # 获取配置中的图像高度 h 和宽度 w, 必须能狗被2的6次方整除, 上采样时需要  w:1024  h：1024
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")  # 检查图像的尺寸是否是 2 的 6 次方的倍数，如果不是，则抛出异常

    # 定义 Input(shape=[...]) -> shape=(?,...)  其中, ?第一维（通常是批量大小）
    # 定义输入图像：输入进来的图片必须是2的6次方以上的倍数(后续还原原图时用到)
    # Tensor("input_image:0", shape=(?, ?, ?, 3), dtype=float32)  shape=(n,c,h,w)
    input_image = Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    # 定义输入图像元数据：meta包含了一些必要信息
    # Tensor("input_image_meta:0", shape=(?, 93), dtype=float32)  1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES(81)
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
    # 定义先验框：输入进来的先验框
    # None 表示输入张量的第一维（通常是批量大小）可以是任意大小,4 表示输入张量的第二维的大小为 4, 这种灵活性允许模型处理不同大小的批量输入。在训练或预测时，可以根据实际情况动态地调整批量大小，而不需要固定为某个特定的值。
    # Tensor("input_anchors:0", shape=(?, ?, 4), dtype=float32)
    input_anchors = Input(shape=[None, 4], name="input_anchors")

    # --------------------------------------------------#
    #  【网络结构-1.卷积部分CNN】 Resnet101 + 特征金字塔FPN
    # --------------------------------------------------#
    '''
    骨干部分Resnet101：
       获得Resnet101里的压缩程度不同的一些层
       backbone:  Mask-RCNN 使用Resenet101作为主干提取网络，对应着图像中的CNN部分(也可用别的CNN网络)
                  C1:Tensor("max_pooling2d_1/MaxPool:0", shape=(?, ?, ?, 64), dtype=float32)  
                  C2:Tensor("res2c_out/Relu:0", shape=(?, ?, ?, 256), dtype=float32) 
                  C3:Tensor("res3d_out/Relu:0", shape=(?, ?, ?, 512), dtype=float32)
                  C4:Tensor("res4w_out/Relu:0", shape=(?, ?, ?, 1024), dtype=float32) 
                  C5:Tensor("res5c_out/Relu:0", shape=(?, ?, ?, 2048), dtype=float32)
    '''
    _, C2, C3, C4, C5 = get_resnet(input_image, stage5=True, train_bn=config.TRAIN_BN)

    '''
    特征金字塔FPN的构建: 
       对于深度卷积网络,从一个特征层卷积到另一个特征层，大的目标被经过卷的次数远比小的目标多，所以在下一个特征层里，会更多的反应大目标的特点，导致大目标的特点更容易得到保留，小目标的特征点容易被跳过。
       使用特征金字塔网络（FPN）将不同层的输出组合成特征金字塔，为了实现特征多尺度的融合，在Mask R-CNN当中，取出在主干特征提取网络中长宽 压缩了两次C2、三次C3、四次C4、五次C5的结果来进行特征金字塔结构的构造
       
       FPN 融合了底层到高层的 feature maps, 从而充分的利用了提取到的各个阶段的特征(Resnet中的C2-C5):
           p2-p5 是将来用于预测物体的 bbox、 box-regression、 mask
           p2-p6 是用于训练RPN的， 即p6只用于RPN网络中
    '''
    #  P5长宽共压缩了5次
    # Height/32,Width/32,256  Tensor("fpn_c5p5/BiasAdd:0", shape=(?, ?, ?, 256), dtype=float32)
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    # P4长宽共压缩了4次
    # Height/16,Width/16,256
    P4 = Add(name="fpn_p4add")([  # Tensor("fpn_p4add/add:0", shape=(?, ?, ?, 256), dtype=float32)
        UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    # P4长宽共压缩了3次
    # Height/8,Width/8,256  Tensor("fpn_p3add/add:0", shape=(?, ?, ?, 256), dtype=float32)
    P3 = Add(name="fpn_p3add")([
        UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    # P4长宽共压缩了2次
    # Height/4,Width/4,256  Tensor("fpn_p2add/add:0", shape=(?, ?, ?, 256), dtype=float32)
    P2 = Add(name="fpn_p2add")([
        UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])

    # 对特征金字塔的不同层进行卷积操作，以获得相同通道数(256)的特征图
    # 各自进行一次256通道的卷积，此时P2、P3、P4、P5通道数相同
    # Height/4,Width/4,256   Tensor("fpn_p2/BiasAdd:0", shape=(?, ?, ?, 256), dtype=float32)
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    # Height/8,Width/8,256   Tensor("fpn_p3/BiasAdd:0", shape=(?, ?, ?, 256), dtype=float32)
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    # Height/16,Width/16,256 Tensor("fpn_p4/BiasAdd:0", shape=(?, ?, ?, 256), dtype=float32)
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    # Height/32,Width/32,256  Tensor("fpn_p5/BiasAdd:0", shape=(?, ?, ?, 256), dtype=float32)
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # 在建议框网络里面还有一个P6用于获取建议框
    # Height/64,Width/64,256  Tensor("fpn_p6/MaxPool:0", shape=(?, ?, ?, 256), dtype=float32)
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    # 定义 RPN 特征图 P2, P3, P4, P5, P6 可以用于获取建议框(P6只在训练当中使用) -> rpn_feature_maps训练当中使用
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    # 定义 Mask RCNN 特征图P2, P3, P4, P5用于获取mask信息(推理当中使用) -> mrcnn_feature_maps推理当中使用
    mrcnn_feature_maps = [P2, P3, P4, P5]

    # 将输入的先验框作为锚点 Tensor("input_anchors:0", shape=(?, ?, 4), dtype=float32)
    anchors = input_anchors  # Tensor("input_anchors:0", shape=(?, ?, 4), dtype=float32)

    # --------------------------------------------------------------------------------------------------------------------------#
    #  【网络结构-2.RPN部分】 分支1：正负(每个框对于所有类别) + 分支2：偏移量 + ProposalLayer合并分支1、2得到每个点的建议框
    #             RPN分支1    分类结果-对每个分类正负 上面一条通过softmax分类anchors，获得positive和negative分类
    #             RPN分支2    回归结果-偏移量  下面一条用于计算对于计算对于 anchors的bouding box regression偏移量，以获取精确的proposal
    #             Proposal层  负责综合positive anchors 和对应的bouding box regression偏移量获取proposals同时剔除太小和超出边界的proposals
    #   到Proposal这里，相当于完成了 目标定位的功能
    # --------------------------------------------------------------------------------------------------------------------------#
    rpn = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)

    # 初始化 RPN 网络的预测结果
    rpn_class_logits, rpn_class, rpn_bbox = [], [], []

    # 获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
    for p in rpn_feature_maps:
        logits, classes, bbox = rpn([p])  # 对每个 RPN 特征图进行 RPN 网络的预测，获得RPN网络的预测结果  中间结果logits 分支1正负 分支2偏移量
        rpn_class_logits.append(logits)   # 并将结果添加到相应的列表 rpn_class_logits 中 -> 中间结果
        rpn_class.append(classes)  # 并将结果添加到相应的列表 rpn_class 中 -> 分支1正负(对每个类别)
        rpn_bbox.append(bbox)  # 并将结果添加到相应的列表 rpn_bbox 中 -> 分支2偏移量
    # 加到一起
    '''
    Concatenate：沿着指定的维度将多个张量连接在一起
    使用 `Concatenate` 层将 `rpn_class_logits` 沿着轴 1 进行拼接。`Concatenate` 层的作用是将多个输入张量沿着指定的轴进行连接，从而得到一个新的张量。
    `name="rpn_class_logits"` 是给这个层指定的一个名称，以便在模型的可视化或调试中更容易识别和理解。
     举例：
           tensor1 = [[1, 2, 3],      tensor2  = [[7, 8, 9],
                      [4, 5, 6]]                 [10, 11, 12]]
                      
                        [[1, 2, 3],                                          [[1, 2, 3, 7, 8, 9],
                         [4, 5, 6]，                                          [4, 5, 6, 10, 11, 12]]
                         [7, 8, 9],  
                         [10, 11, 12]]
           沿着轴 0 进行拼接”的意思是将多个张量在第1个维度上(行)连接起来            沿着轴 1 进行拼接”的意思是将多个张量在第二个维度上(列)连接起来 
           
     concat: 用法同上, 沿着指定的维度将多个张量连接在一起 , concat(tensors, axis)，其中 tensors 是要连接的张量列表或元组，axis 是指定的连接维度 
             concatenated_tensor = tf.concat([tensor1, tensor2], axis=0)
             
     add: 将两个张量相加, add(x, y)，其中 x 和 y 是要相加的两个张量 
          tf.Tensor(  [[ 8 10 12]
                       [14 16 18]],   shape=(2, 3), dtype=int32)
    '''
    rpn_class_logits = Concatenate(axis=1, name="rpn_class_logits")(rpn_class_logits)
    rpn_class = Concatenate(axis=1, name="rpn_class")(rpn_class)  # Tensor("rpn_class/concat:0", shape=(?, ?, 2), dtype=float32)
    rpn_bbox = Concatenate(axis=1, name="rpn_bbox")(rpn_bbox)  # Tensor("rpn_bbox/concat:0", shape=(?, ?, 4), dtype=float32)

    # 此时获得的rpn_class_logits、rpn_class、rpn_bbox的维度是
    # rpn_class_logits : Batch_size, num_anchors, 2  Tensor("rpn_class_logits/concat:0", shape=(?, ?, 2), dtype=float32)
    # rpn_class : Batch_size, num_anchors, 2  Tensor("rpn_class/concat:0", shape=(?, ?, 2), dtype=float32)
    # rpn_bbox : Batch_size, num_anchors, 4  Tensor("rpn_bbox/concat:0", shape=(?, ?, 4), dtype=float32)
    proposal_count = config.POST_NMS_ROIS_INFERENCE  # 获取建议框的数量 1000

    # Batch_size, proposal_count, 4
    # ProposalLayer部分 对先验框进行解码  使用建议框生成层生成建议框
    # Proposal层  负责综合positive anchors 和对应的bouding box regression偏移量获取proposals, 同时剔除太小和超出边界的proposals
    #             Tensor("ROI/packed_2:0", shape=(1, ?, 4), dtype=float32)
    rpn_rois = ProposalLayer(proposal_count=proposal_count,   # 1000
                             nms_threshold=config.RPN_NMS_THRESHOLD,  # 0.7
                             name="ROI",
                             config=config)([rpn_class, rpn_bbox, anchors])  # Tensor("rpn_class/concat:0", shape=(?, ?, 2), dtype=float32)  Tensor("rpn_bbox/concat:0", shape=(?, ?, 4), dtype=float32) Tensor("input_anchors:0", shape=(?, ?, 4), dtype=float32)

    # ----------------------------------------------------------------------------#
    #   【
    #       网络结构-3 ROIAlign部分 -> 最终的预测框
    #       网络结构-4 FC部分(分类+回归) -> FC1 - 分类  获得classifier的结果
    #                                   FC2 - 回归  获得最终的检测结果
    #                                                                   】
    # ----------------------------------------------------------------------------#
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
        fpn_classifier_graph(rpn_rois,    # Tensor("ROI/packed_2:0", shape=(1, ?, 4), dtype=float32)
                             mrcnn_feature_maps,  # [P2, P3, P4, P5]
                             input_image_meta,  # Tensor("input_image_meta:0", shape=(?, 93), dtype=float32)  1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES(81)
                             config.POOL_SIZE,  # 7
                             config.NUM_CLASSES,  # 81
                             train_bn=config.TRAIN_BN,  # False
                             fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)  # 1024

    # ---------- 以上是和fast-r-cnn类似的部分--------------------------------------------------------

    # 使用检测层对检测结果进行处理
    detections = DetectionLayer(config, name="mrcnn_detection")([rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
    # 获取检测框的坐标
    detection_boxes = Lambda(lambda x: x[..., :4])(detections)

    # --------------------------------------------------------------------------------------------------------------#
    #   【Mask部分】
    #    结果3 mask : ROI Align -> Conv(4个,ppt中2个) -> 上采样到原图(反卷积Conv2DTranspose) -> 1X1卷积得到K个通道(K*M*M)
    # --------------------------------------------------------------------------------------------------------------#
    mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                      input_image_meta,
                                      config.MASK_POOL_SIZE,
                                      config.NUM_CLASSES,
                                      train_bn=config.TRAIN_BN)

    # 作为输出
    model = Model([input_image, input_image_meta, input_anchors],
                  [detections, mrcnn_class, mrcnn_bbox,
                   mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                  name='mask_rcnn')
    return model


def get_train_model(config):
    h, w = config.IMAGE_SHAPE[:2]
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")

    # 输入进来的图片必须是2的6次方以上的倍数
    input_image = Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    # meta包含了一些必要信息
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")

    # RPN建议框网络的真实框信息
    input_rpn_match = Input(
        shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
    input_rpn_bbox = Input(
        shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

    # 种类信息
    input_gt_class_ids = Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)

    # 框的位置信息
    input_gt_boxes = Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)

    # 标准化到0-1之间
    gt_boxes = Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)

    # mask语义分析信息
    # [batch, height, width, MAX_GT_INSTANCES]
    if config.USE_MINI_MASK:
        input_gt_masks = Input(shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                               name="input_gt_masks", dtype=bool)
    else:
        input_gt_masks = Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None], name="input_gt_masks",
                               dtype=bool)

    # 获得Resnet里的压缩程度不同的一些层
    _, C2, C3, C4, C5 = get_resnet(input_image, stage5=True, train_bn=config.TRAIN_BN)

    # 组合成特征金字塔的结构
    # P5长宽共压缩了5次
    # Height/32,Width/32,256
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    # P4长宽共压缩了4次
    # Height/16,Width/16,256
    P4 = Add(name="fpn_p4add")([
        UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    # P4长宽共压缩了3次
    # Height/8,Width/8,256
    P3 = Add(name="fpn_p3add")([
        UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    # P4长宽共压缩了2次
    # Height/4,Width/4,256
    P2 = Add(name="fpn_p2add")([
        UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])

    # 各自进行一次256通道的卷积，此时P2、P3、P4、P5通道数相同
    # Height/4,Width/4,256
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    # Height/8,Width/8,256
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    # Height/16,Width/16,256
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    # Height/32,Width/32,256
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # 在建议框网络里面还有一个P6用于获取建议框
    # Height/64,Width/64,256
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    # P2, P3, P4, P5, P6可以用于获取建议框
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    # P2, P3, P4, P5用于获取mask信息
    mrcnn_feature_maps = [P2, P3, P4, P5]

    anchors = get_anchors(config, config.IMAGE_SHAPE)
    # 拓展anchors的shape，第一个维度拓展为batch_size
    anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
    # 将anchors转化成tensor的形式
    anchors = Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
    # 建立RPN模型
    rpn = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)

    rpn_class_logits, rpn_class, rpn_bbox = [], [], []

    # 获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
    for p in rpn_feature_maps:
        logits, classes, bbox = rpn([p])
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)

    rpn_class_logits = Concatenate(axis=1, name="rpn_class_logits")(rpn_class_logits)
    rpn_class = Concatenate(axis=1, name="rpn_class")(rpn_class)
    rpn_bbox = Concatenate(axis=1, name="rpn_bbox")(rpn_bbox)

    # 此时获得的rpn_class_logits、rpn_class、rpn_bbox的维度是
    # rpn_class_logits : Batch_size, num_anchors, 2
    # rpn_class : Batch_size, num_anchors, 2
    # rpn_bbox : Batch_size, num_anchors, 4
    proposal_count = config.POST_NMS_ROIS_TRAINING

    # Batch_size, proposal_count, 4
    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name="ROI",
        config=config)([rpn_class, rpn_bbox, anchors])

    active_class_ids = Lambda(
        lambda x: parse_image_meta_graph(x)["active_class_ids"]
    )(input_image_meta)

    if not config.USE_RPN_ROIS:
        # 使用外部输入的建议框
        input_rois = Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                           name="input_roi", dtype=np.int32)
        # Normalize coordinates
        target_rois = Lambda(lambda x: norm_boxes_graph(
            x, K.shape(input_image)[1:3]))(input_rois)
    else:
        # 利用预测到的建议框进行下一步的操作
        target_rois = rpn_rois

    """找到建议框的ground_truth
    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)]建议框
    gt_class_ids: [batch, MAX_GT_INSTANCES]每个真实框对应的类
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]真实框的位置
    gt_masks: [batch, height, width, MAX_GT_INSTANCES]真实框的语义分割情况

    Returns: 
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]内部真实存在目标的建议框
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]每个建议框对应的类
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]每个建议框应该有的调整参数
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]每个建议框语义分割情况
    """
    rois, target_class_ids, target_bbox, target_mask = \
        DetectionTargetLayer(config, name="proposal_targets")([
            target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

    # 找到合适的建议框的classifier预测结果
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
        fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                             config.POOL_SIZE, config.NUM_CLASSES,
                             train_bn=config.TRAIN_BN,
                             fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
    # 找到合适的建议框的mask预测结果
    mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                      input_image_meta,
                                      config.MASK_POOL_SIZE,
                                      config.NUM_CLASSES,
                                      train_bn=config.TRAIN_BN)

    output_rois = Lambda(lambda x: x * 1, name="output_rois")(rois)

    # Losses
    rpn_class_loss = Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
        [input_rpn_match, rpn_class_logits])
    rpn_bbox_loss = Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
        [input_rpn_bbox, input_rpn_match, rpn_bbox])
    class_loss = Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
        [target_class_ids, mrcnn_class_logits, active_class_ids])
    bbox_loss = Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
        [target_bbox, target_class_ids, mrcnn_bbox])
    mask_loss = Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
        [target_mask, target_class_ids, mrcnn_mask])

    # Model
    inputs = [input_image, input_image_meta,
              input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]

    if not config.USE_RPN_ROIS:
        inputs.append(input_rois)
    outputs = [rpn_class_logits, rpn_class, rpn_bbox,
               mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
               rpn_rois, output_rois,
               rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
    model = Model(inputs, outputs, name='mask_rcnn')
    return model
