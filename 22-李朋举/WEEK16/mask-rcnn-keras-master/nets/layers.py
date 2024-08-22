import tensorflow as tf
from keras.engine import Layer
import numpy as np
from utils import utils


# ----------------------------------------------------------#
#   Proposal Layer
#   该部分代码用于将先验框转化成建议框
# ----------------------------------------------------------#

# 现了将边界框的调整量应用到先验框上的功能，得到了调整后的边界框坐标。
def apply_box_deltas_graph(boxes, deltas):
    # 计算先验框的中心和宽高
    height = boxes[:, 2] - boxes[:, 0]  # 先验框的高度，通过计算框的上边界坐标减去下边界坐标得到
    width = boxes[:, 3] - boxes[:, 1]  # 先验框的宽度，通过计算框的右边界坐标减去左边界坐标得到
    center_y = boxes[:, 0] + 0.5 * height  # 先验框的中心纵坐标，通过计算上边界坐标加上 0.5 倍的高度得到
    center_x = boxes[:, 1] + 0.5 * width  # 先验框的中心横坐标，通过计算左边界坐标加上 0.5 倍的宽度得到
    # 计算出调整后的先验框的中心和宽高
    center_y += deltas[:, 0] * height  # 调整后的先验框的中心纵坐标，通过将先验框的中心纵坐标加上调整量的第一维（deltas[:, 0]）乘以高度得到。
    center_x += deltas[:, 1] * width  # 调整后的先验框的中心横坐标，通过将先验框的中心横坐标加上调整量的第二维（deltas[:, 1]）乘以宽度得到
    height *= tf.exp(deltas[:, 2])  # 调整后的先验框的高度，通过将先验框的高度乘以指数函数 tf.exp(deltas[:, 2]) 得到
    width *= tf.exp(deltas[:, 3])  # 调整后的先验框的宽度，通过将先验框的宽度乘以指数函数 tf.exp(deltas[:, 3]) 得到
    # 计算左上角和右下角的点的坐标
    y1 = center_y - 0.5 * height  # 调整后的边界框的左上角纵坐标，通过计算中心纵坐标减去 0.5 倍的高度得到
    x1 = center_x - 0.5 * width  # 调整后的边界框的左上角横坐标，通过计算中心横坐标减去 0.5 倍的宽度得到
    y2 = y1 + height  # 调整后的边界框的右下角纵坐标，通过计算中心纵坐标加上 0.5 倍的高度得到
    x2 = x1 + width  # 调整后的边界框的右下角横坐标，通过计算中心横坐标加上 0.5 倍的宽度得到
    # 将调整后的边界框的坐标堆叠在一起，并返回
    # 使用 tf.stack 函数将调整后的边界框的左上角和右下角的坐标堆叠在一起，形成一个形状为 (batch_size, 4) 的张量。
    # axis=1 表示沿着第二维进行堆叠。name="apply_box_deltas_out" 为该操作命名。
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


# 裁剪边界框 clip-修剪  对边界框进行裁剪操作，确保边界框的坐标在指定的裁剪窗口范围内
def clip_boxes_graph(boxes, window):
    """
    boxes： [N, (y1, x1, y2, x2)]  是一个形状为 [N, (y1, x1, y2, x2)] 的张量，表示要裁剪的边界框集合。
    window：[4] in the form y1, x1, y2, x2  是一个形状为 [4] 的张量，表示裁剪窗口的坐标，形式为 y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)  # 使用 tf.split 函数将 window 张量沿着最后一个维度拆分成四个子张量：wy1, wx1, wy2, wx2
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)  # 使用 tf.split 函数将 boxes 张量沿着最后一个维度拆分成四个子张量：y1, x1, y2, x2
    # Clip
    # 对于每个边界框的坐标，进行裁剪操作。使用 tf.maximum 函数确保坐标不超出裁剪窗口的上边界，使用 tf.minimum 函数确保坐标不低于裁剪窗口的下边界
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    # 将裁剪后的坐标重新组合成一个形状为 [N, 4] 的张量，使用 tf.concat 函数将四个子张量沿着最后一个维度连接起来
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    # 设置裁剪后张量的形状为 (clipped.shape[0], 4)，以确保张量的形状正确
    clipped.set_shape((clipped.shape[0], 4))
    # 返回裁剪后的边界框张量
    return clipped


"""
Proposal Layers 按照以下顺序一次处理：
  1.利用变换量对所有的 positive anchors 做 bbox regression回归 ->(近似值)
  2.按照输入的 positive softmax scores 由大到小排序 anchors, 提取钱 pre_nms_topN(6000)个anchors，即提取修正位置后的 positive anchors ->(排序)
  3.对剩余的 positive anchors 进行 NMS ->(剩1)
  4.输出 proposals ->(候选框)
"""
class ProposalLayer(Layer):  # 定义 ProposalLayer 的类，它继承自 Layer 类

    # 通过调用父类的初始化方法来初始化父类(Layer)的属性
    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config  # 配置对象
        self.proposal_count = proposal_count  # 建议框的数量
        self.nms_threshold = nms_threshold  # 非极大抑制的阈值

    '''
    call 方法是在模型中被调用的，具体来说，它是在模型的前向传播过程中被调用的:
         在 Keras 中，模型是由一系列层组成的，当我们将输入数据传递给模型时，模型会按照层的顺序依次调用每个层的 call 方法，从而实现前向传播。
         在这个例子中，ProposalLayer 是一个自定义的层，它实现了非极大抑制（NMS）的功能。当模型在训练或预测时，会将输入数据传递给 ProposalLayer， 
         然后调用 call 方法来执行非极大抑制操作，并返回处理后的结果。通过在模型中定义自定义的层，并实现 call 方法，我们可以灵活地扩展模型的功能，实现各种复杂的操作
    inputs -> [rpn_class, rpn_bbox, anchors]  
              [<tf.Tensor 'rpn_class/concat:0' shape=(?, ?, 2) dtype=float32>, 
              <tf.Tensor 'rpn_bbox/concat:0' shape=(?, ?, 4) dtype=float32>, 
              <tf.Tensor 'input_anchors:0' shape=(?, ?, 4) dtype=float32>]
    '''
    def call(self, inputs):
        # 代表这个先验框内部是否有物体[batch, num_rois, 1], 从输入中获取第一个元素，即先验框内部是否有物体的得分，并将其存储在 scores 变量中。
        #                        [:, :, 1] 的索引操作，选择了该数组的所有行和列，但只选择了第三个维度的第二个元素。
        scores = inputs[0][:, :, 1]  # Tensor("ROI/strided_slice:0", shape=(?, ?), dtype=float32)

        # 代表这个先验框的调整参数[batch, num_rois, 4],  从输入中获取第二个元素，即先验框的调整参数，并将其存储在 deltas 变量中
        deltas = inputs[1]  # Tensor("rpn_bbox/concat:0", shape=(?, ?, 4), dtype=float32)

        '''
        标准化rpn框的数值，提升训练效果:
            在深度学习中，deltas 通常表示某种差异或变化量，例如预测的边界框与真实边界框之间的差异。self.config.RPN_BBOX_STD_DEV 可能是一个预先定义的标准偏差或方差值。
            通过将 deltas 乘以 np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])，实现了对 deltas 的缩放操作。这可以用于调整 deltas 的大小，以适应特定的任务或场景。
        '''
        # RPN_BBOX_STD_DEV:[0.1 0.1 0.2 0.2]，改变数量级, 将调整参数乘以一个常数，以改变其数量级
        # Tensor("ROI/mul:0", shape=(?, ?, 4), dtype=float32)
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])

        # Anchors , 从输入中获取第三个元素，即先验框本身，并将其存储在 anchors 变量中
        anchors = inputs[2]  # Tensor("input_anchors:0", shape=(?, ?, 4), dtype=float32)

        # 筛选出得分前6000个的框 (使用 tf.nn.top_k 函数筛选出得分前 pre_nms_limit 个的框的索引，并将其存储在 ix 变量中)
        '''
        确定在进行非极大值抑制（Non-Maximum Suppression，NMS）之前，要保留的候选框的数量。        
            - `self.config.PRE_NMS_LIMIT`：这是一个配置参数，表示在进行 NMS 之前，希望保留的候选框的最大数量。 -> 6000
            - `tf.shape(anchors)[1]`：这是计算张量 `anchors` 的第二个维度的大小，即候选框的数量。
            - `tf.minimum(...)`：这是 TensorFlow 中的一个函数，用于返回两个参数中的最小值。
            综上所述，`pre_nms_limit` 变量将被赋值为 `self.config.PRE_NMS_LIMIT` 和 `tf.shape(anchors)[1]` 中的较小值。这意味着在进行 NMS 之前，将只保留数量不超过 `pre_nms_limit` 的候选框。
        这样做的原因是为了在进行 NMS 之前，减少需要处理的候选框数量，从而提高计算效率。同时，通过设置 `self.config.PRE_NMS_LIMIT`，可以控制保留的候选框数量，以平衡计算效率和检测性能。
        '''
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])  # Tensor("ROI/Minimum:0", shape=(), dtype=int32)

        # 获得这些框的索引
        '''
        使用 TensorFlow 的 `nn.top_k` 函数来获取得分（`scores`）最高的前 `pre_nms_limit` 个索引（`indices`）。        
            - `tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors")`：这是 TensorFlow 提供的 `top_k` 函数，用于从输入的张量 `scores` 中找出前 `pre_nms_limit` 个最大值的索引。
                - `scores`：这是一个张量，表示每个候选框的得分。
                - `pre_nms_limit`：这是一个整数，表示要获取的前 `k` 个索引的数量。
                - `sorted=True`：表示返回的索引是按照得分降序排列的。
                - `name="top_anchors"`：这是给操作命名，以便在 TensorBoard 或其他调试工具中进行可视化和跟踪。
            - `.indices`：这是 `top_k` 操作返回的结果，它是一个张量，包含了前 `pre_nms_limit` 个最大值的索引。
        综上所述，作用是获取得分最高的前 `pre_nms_limit` 个候选框的索引，并将其存储在 `ix` 变量中。这些索引可以用于后续的处理，例如在非极大值抑制（NMS）中选择要保留的候选框。
        '''
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices  # Tensor("ROI/top_anchors:1", shape=(?, ?), dtype=int32)

        """
        `tf.gather` 是 TensorFlow 中的一个函数，用于根据索引从张量中收集元素:
            ```python
            tf.gather(params, indices, validate_indices=None, axis=None, batch_dims=0, name=None)
            ```
            参数说明：
            - `params`：要收集元素的张量。
            - `indices`：指定要收集的元素的索引张量。索引可以是整数或整数张量。
            - `validate_indices`：可选参数，用于检查索引是否有效。如果设置为 `True`，则会检查索引是否在 `params` 的范围内。
            - `axis`：可选参数，指定沿着哪个轴进行收集。默认值为 `None`，表示在整个张量中收集元素。
            - `batch_dims`：可选参数，指定要保留的批量维度的数量。默认值为 `0`，表示不保留批量维度。
            - `name`：可选参数，为操作指定名称。
            返回值：
            `tf.gather` 函数返回一个与 `indices` 形状相同的张量，其中包含从 `params` 中根据索引收集的元素。
        """

        # 获得这些框的得分 (使用 utils.batch_slice 函数和 tf.gather 函数，根据索引 ix 从得分中获取筛选出的框的得分，并将其存储在 scores 变量中)
        '''
        使用`utils.batch_slice` 函数对输入的 `scores` 和 `ix` 进行了切片操作，并使用 `tf.gather` 函数对切片后的结果进行了收集。        
            1. `utils.batch_slice` 函数：这是一个工具函数，用于对输入的张量进行切片操作。它接受两个参数：`inputs` 和 `slice_fn`。
                - `inputs`：这是一个列表或元组，包含了要切片的张量。
                - `slice_fn`：这是一个函数，用于对每个切片进行处理。它接受两个参数：`x` 和 `y`，分别表示要处理的张量和切片的索引。
            2. `lambda x, y: tf.gather(x, y)`：这是一个匿名函数，作为 `slice_fn` 参数传递给 `utils.batch_slice` 函数。
                             它的作用是使用 `tf.gather` 函数从输入的张量 `x` 中根据索引 `y` 收集切片后的结果。
            3. `self.config.IMAGES_PER_GPU`：这是一个配置参数，表示每个 GPU 要处理的图像数量。
        综上所述，这段代码的作用是对输入的 `scores` 和 `ix` 进行切片操作，将每个 GPU 要处理的图像对应的得分收集起来。这样可以方便地对每个 GPU 上的结果进行后续处理。
        '''
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)

        # 获得这些框的调整参数 (使用 utils.batch_slice 函数和 tf.gather 函数，根据索引 ix 从调整参数中获取筛选出的框的调整参数，并将其存储在 deltas 变量中)
        '''
        这段代码使用了 `utils.batch_slice` 函数对输入的 `deltas` 和 `ix` 进行了切片操作，并使用 `tf.gather` 函数对切片后的结果进行了收集。            
            1. `utils.batch_slice` 函数：这是一个工具函数，用于对输入的张量进行切片操作。它接受两个参数：`inputs` 和 `slice_fn`。
                - `inputs`：这是一个列表或元组，包含了要切片的张量。
                - `slice_fn`：这是一个函数，用于对每个切片进行处理。它接受两个参数：`x` 和 `y`，分别表示要处理的张量和切片的索引。
            2. `lambda x, y: tf.gather(x, y)`：这是一个匿名函数，作为 `slice_fn` 参数传递给 `utils.batch_slice` 函数。它的作用是使用 `tf.gather` 函数从输入的张量 `x` 中根据索引 `y` 收集切片后的结果。
            3. `self.config.IMAGES_PER_GPU`：这是一个配置参数，表示每个 GPU 要处理的图像数量。
        综上所述，这段代码的作用是对输入的 `deltas` 和 `ix` 进行切片操作，将每个 GPU 要处理的图像对应的 `deltas` 收集起来。这样可以方便地对每个 GPU 上的结果进行后续处理。
        '''
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)

        # 获得这些框对应的先验框 (使用 utils.batch_slice 函数和 tf.gather 函数，根据索引 ix 从先验框中获取筛选出的框对应的先验框，并将其存储在 pre_nms_anchors 变量中)
        '''
        1. `anchors`：这是一个张量，表示所有的锚框（anchors）。
        2. `ix`：这是一个张量，表示要选择的锚框的索引。
        3. `utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.config.IMAGES_PER_GPU, names=["pre_nms_anchors"])`：这是一个使用 `utils.batch_slice` 函数的操作。
            - `[anchors, ix]`：这是要切片的张量列表，包含了锚框和索引。
            - `lambda a, x: tf.gather(a, x)`：这是一个匿名函数，作为切片函数。它接受两个参数 `a` 和 `x`，分别表示锚框和索引。在函数内部，使用 `tf.gather` 函数根据索引 `x` 从锚框张量 `a` 中选择相应的锚框。
            - `self.config.IMAGES_PER_GPU`：这是每个 GPU 要处理的图像数量。
            - `names=["pre_nms_anchors"]`：这是给切片操作命名，以便在后续的代码中更容易识别和引用。
        综上所述，这段代码的作用是根据索引 `ix` 从锚框张量 `anchors` 中选择一部分锚框，并将结果存储在 `pre_nms_anchors` 张量中。这样可以在每个 GPU 上处理一部分锚框，以便进行后续的计算。
        '''
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.config.IMAGES_PER_GPU, names=["pre_nms_anchors"])

        # [batch, N, (y1, x1, y2, x2)]
        # 对先验框进行解码 (使用 utils.batch_slice 函数和 apply_box_deltas_graph 函数，对筛选出的框的先验框进行解码，得到最终的框的坐标，并将其存储在 boxes 变量中)
        '''
        1. `pre_nms_anchors` 和 `deltas` 是两个输入张量，它们可能表示锚框和对应的偏移量。
        2. `utils.batch_slice` 函数用于对输入的张量进行切片操作。它接受以下参数：
            - `inputs`：要切片的张量列表，这里是 `[pre_nms_anchors, deltas]`。
            - `slice_fn`：切片函数，用于对每个切片进行处理。这里使用了一个匿名函数 `lambda x, y: apply_box_deltas_graph(x, y)`，它将对每个切片应用 `apply_box_deltas_graph` 函数。
            - `IMAGES_PER_GPU`：每个 GPU 要处理的图像数量。
            - `names`：切片操作的名称列表，这里是 `["refined_anchors"]`。
        3. 在切片函数中，`apply_box_deltas_graph` 函数可能用于根据锚框和偏移量计算得到调整后的框的坐标。
        4. 最终，代码将返回一个包含调整后框的坐标的张量 `boxes`。
        总结起来，这段代码的目的是对锚框和偏移量进行切片操作，并通过应用 `apply_box_deltas_graph` 函数计算得到调整后的框的坐标。具体的计算过程和结果取决于 `apply_box_deltas_graph` 函数的实现。
        '''
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # [batch, N, (y1, x1, y2, x2)]
        # 防止超出图片范围 (使用 utils.batch_slice 函数和 clip_boxes_graph 函数，对解码后的框进行裁剪，以防止超出图片范围，并将其存储在 boxes 变量中)
        '''
        对解码后的框进行裁剪，以防止超出图片范围:
            1. `window = np.array([0, 0, 1, 1], dtype=np.float32)`: 创建一个表示窗口的 NumPy 数组，其中 `[0, 0, 1, 1]` 表示窗口的左上角坐标为 `(0, 0)`，右下角坐标为 `(1, 1)`。这个窗口通常用于定义图片的有效区域。
            2. `utils.batch_slice(boxes,...)`: 使用 `utils.batch_slice` 函数对 `boxes` 进行切片操作。
                - `boxes`: 要切片的张量，可能是解码后的框的坐标。
                - `lambda x: clip_boxes_graph(x, window)`: 切片函数，它接受一个参数 `x`，并将其传递给 `clip_boxes_graph` 函数，同时传递窗口参数 `window`。
                - `self.config.IMAGES_PER_GPU`: 每个 GPU 要处理的图像数量。
                - `names=["refined_anchors_clipped"]`: 切片操作的名称。
            3. `clip_boxes_graph(x, window)`: 这是一个函数，用于对框进行裁剪。它接受框的坐标 `x` 和窗口参数 `window`，并返回裁剪后的框的坐标。
        综上所述，这段代码的作用是将解码后的框的坐标进行切片，并对每个切片应用裁剪函数，以确保框的坐标不会超出图片的有效区域。最后，将裁剪后的框的坐标存储在 `boxes` 变量中。这样可以保证后续的处理只考虑在图片范围内的框。
        '''
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # 非极大抑制
        # 定义了一个非极大抑制的函数 nms，用于对框进行非极大抑制。然后，使用 utils.batch_slice 函数对框和得分进行切片，并应用非极大抑制函数，得到最终的建议框，并将其存储在 proposals 变量中
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # 如果数量达不到设置的建议框数量的话
            # 就padding
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([boxes, scores], nms, self.config.IMAGES_PER_GPU)
        return proposals  # 返回最终的建议框

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


# ----------------------------------------------------------#
#   ROIAlign Layer
#   利用建议框在特征层上截取内容
# ----------------------------------------------------------#

def log2_graph(x):
    return tf.log(x) / tf.log(2.0)


def parse_image_meta_graph(meta):
    """
    将meta里面的参数进行分割
    """
    image_id = meta[:, 0]  # 图像的唯一标识符
    original_image_shape = meta[:, 1:4]  # 原始图像的形状
    image_shape = meta[:, 4:7]  # 当前图像的形状
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels  图像在窗口中的位置（以像素为单位）
    scale = meta[:, 11]  # 图像的缩放比例
    active_class_ids = meta[:, 12:]  # 活动类别的标识符
    # 将提取的参数以字典的形式返回
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }

'''

'''
class PyramidROIAlign(Layer):  # 定义 PyramidROIAlign 的类, 继承自 Layer 类
    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):

        # 接收输入的参数 inputs，并将其分解为三个部分：boxes（建议框的位置）、image_meta（包含图片信息）和 feature_maps（所有的特征层）
        # 建议框的位置
        boxes = inputs[0]
        # image_meta包含了一些必要的图片信息
        image_meta = inputs[1]
        # 取出所有的特征层[batch, height, width, channels]
        feature_maps = inputs[2:]

        # 通过 tf.split 函数将 boxes 按照维度 2 拆分成四个部分，分别赋值给变量 y1、x1、y2 和 x2。然后计算建议框的高度 h 和宽度 w
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        # 获得输入进来的图像的大小
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]

        # 通过建议框的大小找到这个建议框属于哪个特征层
        '''
        计算建议框的面积 image_area，然后通过计算平方根和比例的对数，找到建议框所属的特征层 roi_level。
        最后，通过 tf.minimum 和 tf.maximum 函数对 roi_level 进行限制，确保其在有效范围内。
        '''
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        # batch_size, box_num
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        '''
        通过循环遍历特征层，对于每个特征层，找到对应的建议框，并使用 tf.image.crop_and_resize 函数进行截取和调整大小。将截取后的结果添加到列表 pooled 中
        '''
        pooled = []
        box_to_level = []
        # 分别在P2-P5中进行截取
        for i, level in enumerate(range(2, 6)):
            # 找到每个特征层对应box
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)
            box_to_level.append(ix)

            # 获得这些box所属的图片
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # 停止梯度下降
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(feature_maps[i], level_boxes, box_indices, self.pool_shape, method="bilinear"))

        pooled = tf.concat(pooled, axis=0)

        '''
        将 box_to_level 进行堆叠和扩展，然后根据索引进行排序，将同一张图里的建议框聚集在一起。最后，根据排序后的索引获取图片的索引，并从 pooled 中选择相应的部分
        '''
        # 将顺序和所属的图片进行堆叠
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # box_to_level[:, 0]表示第几张图
        # box_to_level[:, 1]表示第几张图里的第几个框
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        # 进行排序，将同一张图里的某一些聚集在一起
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]

        # 按顺序获得图片的索引
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # 重新reshape为原来的格式
        # 也就是 将截取和调整大小后的结果重新调整为原始的格式，返回最终的输出
        # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


# ----------------------------------------------------------#
#   Detection Layer
#   
# ----------------------------------------------------------#

def refine_detections_graph(rois, probs, deltas, window, config):
    """细化分类建议并过滤重叠部分并返回最终结果探测。
    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # 找到得分最高的类
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # 序号+类
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    # 取出成绩
    class_scores = tf.gather_nd(probs, indices)
    # 还有框的调整参数
    deltas_specific = tf.gather_nd(deltas, indices)
    # 进行解码
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # 防止超出0-1
    refined_rois = clip_boxes_graph(refined_rois, window)

    # 去除背景
    keep = tf.where(class_ids > 0)[:, 0]
    # 去除背景和得分小的区域
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # 获得除去背景并且得分较高的框还有种类与得分
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]

        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)

        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))

        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)

        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. 进行非极大抑制
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. 找到符合要求的需要被保留的建议框
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]

    # 寻找得分最高的num_keep个框
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # 如果达不到数量的话就padding
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


'''
定义 `norm_boxes_graph` 的函数，用于将边界框进行归一化处理:
    1. `boxes`：表示要归一化的边界框张量，形状为 `[..., 4]`，其中 `...` 表示任意数量的前导维度。
    2. `shape`：表示图像的形状张量，形状为 `[2]`，其中包含图像的高度 `h` 和宽度 `w`。
    函数的主要步骤如下：
    1. 将 `shape` 张量转换为 `float32` 类型，并使用 `tf.split` 函数将其拆分为高度 `h` 和宽度 `w`。
    2. 计算归一化的比例因子 `scale`，，通过将高度和宽度连接起来并减去常量 `1.0`。
    3. 计算归一化的偏移量 `shift`，使用常量 `[0., 0., 1., 1.]`。
    4. 最后，通过将边界框减去偏移量，并除以比例因子，实现边界框的归一化。
归一化的目的是将边界框的坐标表示为相对于图像大小的相对值，以便在不同大小的图像上进行一致的处理和比较。
总结来说，该函数的作用是将边界框的坐标进行归一化，使其表示为相对于图像大小的相对位置。这样可以方便地在不同尺寸的图像上进行后续的处理和分析。
'''
def norm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


# 定义了一个名为 DetectionLayer 的类，它继承自 Layer 类。在类的构造函数中，接收一个参数 config，并将其赋值给实例变量 self.config
# 在每个图像上运行检测 refinement 图，并将结果重新整形为指定的形状。
class DetectionLayer(Layer):

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    # 在 call 方法中，接收输入的参数 inputs，并将其分解为四个部分：rois（感兴趣区域）、mrcnn_class（M-RCNN 类别）、mrcnn_bbox（M-RCNN 边界框）和 image_meta（图像元数据）。
    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # 找到window的小数形式: 使用 parse_image_meta_graph 函数解析 image_meta，获取图像的形状信息，并将其赋值给变量 image_shape。
        #                     然后，使用 norm_boxes_graph 函数将 m['window'] 转换为小数形式的窗口，并将其赋值给变量 window。
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        '''
        使用 utils.batch_slice 函数对输入的参数进行切片，并对每个切片应用 refine_detections_graph 函数进行处理。
            self.config.IMAGES_PER_GPU 用于指定每个 GPU 处理的图像数量。
        '''
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        '''
        将处理后的结果进行重新整形，将其转换为 [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6] 的形状，并返回。
        '''
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


# ----------------------------------------------------------#
#   Detection Target Layer
#   该部分代码会输入建议框
#   判断建议框和真实框的重合情况
#   筛选出内部包含物体的建议框
#   利用建议框和真实框编码
#   调整mask的格式使得其和预测格式相同
# ----------------------------------------------------------#

def overlaps_graph(boxes1, boxes2):
    """
    用于计算boxes1和boxes2的重合程度
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    返回 [len(boxes1), len(boxes2)]
    """
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # 移除之前获得的padding的部分
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # 计算建议框和所有真实框的重合程度 [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # 计算和 crowd boxes 的重合程度 [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. 正样本建议框和真实框的重合程度大于0.5
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. 负样本建议框和真实框的重合程度小于0.5，Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # 进行正负样本的平衡
    # 取出最大33%的正样本
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # 保持正负样本比例
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # 获得正样本和负样本
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # 获取建议框和真实框重合程度
    positive_overlaps = tf.gather(overlaps, positive_indices)

    # 判断是否有真实框
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    # 找到每一个建议框对应的真实框和种类
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # 解码获得网络应该有得预测结果
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # 切换mask的形式[N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)

    # 取出对应的层
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # 防止resize后的结果不是1或者0
    masks = tf.round(masks)

    # 一般传入config.TRAIN_ROIS_PER_IMAGE个建议框进行训练，
    # 如果数量不够则padding
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


def trim_zeros_graph(boxes, name='trim_zeros'):
    """
    如果前一步没有满POST_NMS_ROIS_TRAINING个建议框，会有padding
    要去掉padding
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


class DetectionTargetLayer(Layer):
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

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # 对真实框进行编码
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]
