import numpy as np
import math
from utils.utils import norm_boxes


#------------------------------------------------------------------------------------------------------------#
#  Anchors：
#     根据给定的尺度、比例、特征图形状和锚框步长，生成多尺度和多比例的先验框（anchors）
#     函数的输入参数包括尺度 scales、比例 ratios、特征图的形状 shape、特征步长 feature_stride 和锚框步长 anchor_stride。
#------------------------------------------------------------------------------------------------------------#

# scales->32  ratios->[0.5, 1, 2]  shape->[256,256]  feature_stride->4  anchor_stride->1
def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    # 获得所有框的长度和比例的组合
    # scales 和 ratios 是两个列表或数组，表示要生成的先验框的尺度和比例。通过 np.meshgrid 函数将它们转换为网格形式，以便进行组合
    '''
    将输入的两个数组 `scales` 和 `ratios` 转换为网格形式:
        具体来说，`np.meshgrid` 函数会将两个数组作为参数，并返回两个多维数组。第一个多维数组的维度与 `scales` 相同，第二个多维数组的维度与 `ratios` 相同。
        这两个多维数组的元素对应了 `scales` 和 `ratios` 中所有元素的组合。

        在代码中，`scales` 和 `ratios` 被转换为网格形式后，通过 `flatten` 方法将它们展平为一维数组，以便后续的计算。

        为了方便地对 `scales` 和 `ratios` 中的元素进行组合和操作，例如在后续的代码中计算每个组合对应的先验框的高度和宽度。

    '''
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    # scales 被展平为一维数组
    scales = scales.flatten()
    # ratios 被展平为一维数组
    ratios = ratios.flatten()
    # heights 和 widths 分别计算每个尺度和比例组合下的先验框的高度和宽度
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # 生成网格中心
    # 通过 np.meshgrid 函数将 shifts_x 和 shifts_y 转换为网格形式，得到每个网格点的坐标
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    # 根据特征图的形状和锚框步长生成的网格中心的偏移量
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # 获得先验框的中心和宽高 分别计算每个网格点的先验框的宽度、高度和中心坐标
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # 更变格式  box_centers 和 box_sizes 分别将中心坐标和宽高组合成先验框的坐标表示形式
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # 计算出(y1, x1, y2, x2)  将中心坐标和宽高组合成先验框的坐标表示形式 (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
    # 函数返回生成的先验框的坐标
    return boxes


'''
定义了一个名为 `generate_pyramid_anchors` 的函数，用于生成不同特征层的锚框（anchors），并将它们堆叠在一起:
    根据不同的特征层参数生成锚框，并将它们组合成一个统一的锚框数组，以便在目标检测等任务中使用。这些锚框可以用于在不同的特征层上检测目标的位置和大小。
'''
def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """
    生成不同特征层的anchors，并利用concatenate进行堆叠
    """
    # Anchors
    # scales: (32, 64, 128, 256, 512)  不同特征层的尺度值 :P2对应scale是32、P3对应scale是64、P4对应scale是128、P5对应scale是256、P6对应scale是512
    # ratios: [0.5, 1, 2]   包含了锚框的宽高比
    # feature_shapes:  [[256 256], [128 128], [ 64  64], [ 32  32], [ 16  16]]  不同特征层的特征形状,每个特征层（P2、P3、P4、P5、P6）的宽度和高度
    # feature_strides： [4, 8, 16, 32, 64] 不同特征层的特征步长
    # anchor_stride:1 锚框的步长
    # [anchor_count, (y1, x1, y2, x2)]

    # 定义空列表，生成的锚框被添加到 `anchors` 列表中
    anchors = []
    # 循环遍历不同的特征层
    for i in range(len(scales)):
        # 调用 `generate_anchors` 函数生成该特征层的锚框
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride))
    # 将所有特征层的锚框堆叠在一起，形成一个大的锚框数组
    return np.concatenate(anchors, axis=0)

'''
定义了一个名为 `compute_backbone_shapes` 的函数，用于计算主干特征提取网络的形状。
    1. `config`：这是一个配置类对象，可能包含了关于主干网络的各种设置，例如网络类型、步长等。   
    2. `image_shape`：这是输入图像的形状，通常是一个元组，表示图像的高度、宽度和通道数。   (1024, 1024, 3)
    3. 函数首先检查 `config.BACKBONE` 是否可调用。如果是，则使用 `config.COMPUTE_BACKBONE_SHAPE(image_shape)` 来计算主干网络的形状。
    4. 如果 `config.BACKBONE` 不可调用，则断言 `config.BACKBONE` 必须是 `["resnet50", "resnet101"]` 中的一个。这意味着主干网络是一个预定义的 ResNet 模型。
    5. 对于预定义的 ResNet 模型，函数使用 `config.BACKBONE_STRIDES` 中的步长来计算每个特征层（P2、P3、P4、P5、P6）的宽度和高度
    6. 最后，函数返回一个 NumPy 数组，其中包含了每个特征层的宽度和高度。
总的来说，这个函数的目的是根据配置和输入图像的形状，计算主干特征提取网络中各个特征层的形状。这些形状信息在后续的目标检测或其他任务中可能会被用到。
'''
def compute_backbone_shapes(config, image_shape):
    # 用于计算主干特征提取网络的shape
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)
    # 其实就是计算P2、P3、P4、P5、P6这些特征层的宽和高
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])


def get_anchors(config, image_shape):
    # 计算 backbone 网络的输出形状  计算每个特征层（P2、P3、P4、P5、P6）的宽度和高度  [[256 256], [128 128], [ 64  64], [ 32  32], [ 16  16]]
    backbone_shapes = compute_backbone_shapes(config, image_shape)
    # 创建一个空的锚点缓存字典
    anchor_cache = {}
    # 如果图像形状不在锚点缓存中
    if not tuple(image_shape) in anchor_cache:
        # 使用 generate_pyramid_anchors 函数生成金字塔锚点
        a = generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,  # (32, 64, 128, 256, 512)
            config.RPN_ANCHOR_RATIOS,  # [0.5, 1, 2]
            backbone_shapes,
            config.BACKBONE_STRIDES,  # [4, 8, 16, 32, 64]
            config.RPN_ANCHOR_STRIDE)  # 1
        anchor_cache[tuple(image_shape)] = norm_boxes(a, image_shape[:2])
    return anchor_cache[tuple(image_shape)]


