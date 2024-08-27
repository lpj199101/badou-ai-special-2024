import tensorflow as tf
from keras.engine import Layer
import numpy as np
from utils import utils


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#   Proposal Layer
#   该部分代码用于将先验框转化成建议框
#         apply_box_deltas_graph 函数用于根据给定的先验框（boxes）和调整量（deltas）计算调整后的边界框的坐标,它通过对先验框的中心坐标、宽度和高度进行调整来得到新的边界框。
#              具体来说，代码首先计算了先验框的高度、宽度和中心坐标。然后，根据调整量对中心坐标进行调整，并通过指数函数对高度和宽度进行缩放。
#              最后，计算出调整后的边界框的左上角和右下角坐标，并将它们堆叠在一起返回。
#         整个过程中没有涉及到最邻近插值算法，而是基于先验框和调整量进行的坐标计算和变换
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# 将边界框的调整量应用到先验框上的功能，得到了调整后的边界框坐标
# boxes -> Tensor("ROI/strided_slice_8:0", shape=(?, 4), dtype=float32)
# deltas -> Tensor("ROI/strided_slice_9:0", shape=(?, 4), dtype=float32)
def apply_box_deltas_graph(boxes, deltas):
    # 计算先验框的中心和宽高  height：计算了 boxes 张量中每一行的第二个元素减去第一个元素的结果，并将其存储在 height 张量中
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
    clipped = tf.concat([y1, x1, y2, x2], axis=1,
                        name="clipped_boxes")  # Tensor("ROI/clipped_boxes:0", shape=(?, 4), dtype=float32)
    # 设置裁剪后张量的形状为 (clipped.shape[0], 4)，以确保张量的形状正确, 张量 clipped 的形状将从原来的维度变为 (行数, 4) 的形状
    clipped.set_shape((clipped.shape[0], 4))  # Tensor("ROI/clipped_boxes:0", shape=(?, 4), dtype=float32)
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
        标准化rpn框的数值，提升训练效果: (`deltas`) * (一个形状为 `(1, 1, 4)` 的数组 `self.config.RPN_BBOX_STD_DEV` 的重塑版本)
              deltas 可能是一个与边界框调整相关的数组, 通常表示某种差异或变化量，例如预测的边界框与真实边界框之间的差异, self.config.RPN_BBOX_STD_DEV 可能是一个预先定义的标准偏差或方差值,
              通过乘以 `self.config.RPN_BBOX_STD_DEV`，可以对 `deltas` 进行某种标准化或缩放操作
                    例如，如果 `self.config.RPN_BBOX_STD_DEV` 中的值较大，那么乘以它会使 `deltas` 变大，从而增加边界框的调整幅度；
                         反之，如果 `self.config.RPN_BBOX_STD_DEV` 中的值较小，那么乘以它会使 `deltas` 变小，从而减小边界框的调整幅度。
        这样的操作通常在目标检测或其他与边界框相关的任务中使用，以根据特定的配置或标准对边界框的调整进行控制。                
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
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT,
                                   tf.shape(anchors)[1])  # Tensor("ROI/Minimum:0", shape=(), dtype=int32)

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
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices  # Tensor("ROI/top_anchors:1", shape=(?, ?), dtype=int32)

        """
        `tf.gather` 是 TensorFlow 中的一个函数，用于根据索引从张量中收集元素:
            ```python
            tf.gather(params, indices)
            ```
            参数说明：
            - `params`：要收集元素的张量。
            - `indices`：指定要收集的元素的索引张量。索引可以是整数或整数张量。
            返回值：
            `tf.gather` 函数返回一个与 `indices` 形状相同的张量，其中包含从 `params` 中根据索引收集的元素。
        """

        # 获得这些框的得分 (使用 utils.batch_slice 函数和 tf.gather 函数，根据索引 ix 从得分中获取筛选出的框的得分，并将其存储在 scores 变量中)
        '''
        使用`utils.batch_slice` 函数对输入的 `scores` 和 `ix` 进行了切片操作，并使用 `tf.gather` 函数对切片后的结果进行了收集。        
            1. `utils.batch_slice` 函数：这是一个工具函数，用于对输入的张量进行切片操作。它接受两个参数：`inputs` 和 `slice_fn`。
                - `inputs`：这是一个列表或元组，包含了要切片的张量
                           [scores, ix] [ Tensor("ROI/strided_slice:0", shape=(?, ?), dtype=float32), Tensor("ROI/top_anchors:1", shape=(?, ?), dtype=int32)]
                - `slice_fn`：这是一个函数，用于对每个切片进行处理。它接受两个参数：`x` 和 `y`，分别表示要处理的张量和切片的索引。
            2. `lambda x, y: tf.gather(x, y)`：这是一个匿名函数，作为 `slice_fn` 参数传递给 `utils.batch_slice` 函数。
                             它的作用是使用 `tf.gather` 函数从输入的张量 `x` 中根据索引 `y` 收集切片后的结果。
            3. `self.config.IMAGES_PER_GPU`：这是一个配置参数，表示每个 GPU 要处理的图像数量。
        综上所述，这段代码的作用是对输入的 `scores` 和 `ix` 进行切片操作，将每个 GPU 要处理的图像对应的得分收集起来。这样可以方便地对每个 GPU 上的结果进行后续处理。
        '''
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)  # Tensor("ROI/packed:0", shape=(1, ?), dtype=float32)

        # 获得这些框的调整参数 (使用 utils.batch_slice 函数和 tf.gather 函数，根据索引 ix 从调整参数中获取筛选出的框的调整参数，并将其存储在 deltas 变量中)
        '''
        这段代码使用了 `utils.batch_slice` 函数对输入的 `deltas` 和 `ix` 进行了切片操作，并使用 `tf.gather` 函数对切片后的结果进行了收集。            
            1. `utils.batch_slice` 函数：这是一个工具函数，用于对输入的张量进行切片操作。它接受两个参数：`inputs` 和 `slice_fn`。
                - `inputs`：这是一个列表或元组，包含了要切片的张量
                          [Tensor("ROI/mul:0", shape=(?, ?, 4), dtype=float32) ,Tensor("ROI/top_anchors:1", shape=(?, ?), dtype=int32)]
                - `slice_fn`：这是一个函数，用于对每个切片进行处理。它接受两个参数：`x` 和 `y`，分别表示要处理的张量和切片的索引。
            2. `lambda x, y: tf.gather(x, y)`：这是一个匿名函数，作为 `slice_fn` 参数传递给 `utils.batch_slice` 函数。它的作用是使用 `tf.gather` 函数从输入的张量 `x` 中根据索引 `y` 收集切片后的结果。
            3. `self.config.IMAGES_PER_GPU`：这是一个配置参数，表示每个 GPU 要处理的图像数量。
        综上所述，这段代码的作用是对输入的 `deltas` 和 `ix` 进行切片操作，将每个 GPU 要处理的图像对应的 `deltas` 收集起来。这样可以方便地对每个 GPU 上的结果进行后续处理。
        '''
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)  # Tensor("ROI/packed_1:0", shape=(1, ?, 4), dtype=float32)

        # 获得这些框对应的先验框 (使用 utils.batch_slice 函数和 tf.gather 函数，根据索引 ix 从先验框中获取筛选出的框对应的先验框，并将其存储在 pre_nms_anchors 变量中)
        '''
        1. `anchors`：这是一个张量，表示所有的锚框（anchors）。
        2. `ix`：这是一个张量，表示要选择的锚框的索引。
        3. `utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.config.IMAGES_PER_GPU, names=["pre_nms_anchors"])`：这是一个使用 `utils.batch_slice` 函数的操作。
            - `[anchors, ix]`：这是要切片的张量列表，包含了锚框和索引
                               [Tensor("input_anchors:0", shape=(?, ?, 4), dtype=float32), Tensor("ROI/top_anchors:1", shape=(?, ?), dtype=int32)]
            - `lambda a, x: tf.gather(a, x)`：这是一个匿名函数，作为切片函数。它接受两个参数 `a` 和 `x`，分别表示锚框和索引。在函数内部，使用 `tf.gather` 函数根据索引 `x` 从锚框张量 `a` 中选择相应的锚框。
            - `self.config.IMAGES_PER_GPU`：这是每个 GPU 要处理的图像数量。
            - `names=["pre_nms_anchors"]`：这是给切片操作命名，以便在后续的代码中更容易识别和引用。
        综上所述，这段代码的作用是根据索引 `ix` 从锚框张量 `anchors` 中选择一部分锚框，并将结果存储在 `pre_nms_anchors` 张量中。这样可以在每个 GPU 上处理一部分锚框，以便进行后续的计算。
        '''
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.config.IMAGES_PER_GPU,
                                            names=["pre_nms_anchors"])  # Tensor("ROI/pre_nms_anchors:0", shape=(1, ?, 4), dtype=float32)

        # [batch, N, (y1, x1, y2, x2)]
        # 对先验框进行解码 (使用 utils.batch_slice 函数和 apply_box_deltas_graph 函数，对筛选出的框的先验框进行解码，得到最终的框的坐标，并将其存储在 boxes 变量中)
        '''
        1. `pre_nms_anchors` 和 `deltas` 是两个输入张量，它们可能表示锚框和对应的偏移量。
        2. `utils.batch_slice` 函数用于对输入的张量进行切片操作。它接受以下参数：
            - `inputs`：要切片的张量列表，这里是 `[pre_nms_anchors, deltas]`。
                       [Tensor("ROI/pre_nms_anchors:0", shape=(1, ?, 4), dtype=float32),Tensor("ROI/packed_1:0", shape=(1, ?, 4), dtype=float32)]
            - `slice_fn`：切片函数，用于对每个切片进行处理。 `lambda x, y: apply_box_deltas_graph(x, y)`:
                                  它将对每个切片应用 `apply_box_deltas_graph` 函数 ->将边界框的调整量应用到先验框上的功能，得到了调整后的边界框坐标
            - `IMAGES_PER_GPU`：每个 GPU 要处理的图像数量。
            - `names`：切片操作的名称列表，这里是 `["refined_anchors"]`。
        3. 在切片函数中，`apply_box_deltas_graph` 函数可能用于根据锚框和偏移量计算得到调整后的框的坐标。
        4. 最终，代码将返回一个包含调整后框的坐标的张量 `boxes`。
        总结起来，这段代码的目的是对锚框和偏移量进行切片操作，并通过应用 `apply_box_deltas_graph` 函数计算得到调整后的框的坐标。具体的计算过程和结果取决于 `apply_box_deltas_graph` 函数的实现。
        '''
        boxes = utils.batch_slice([pre_nms_anchors, deltas], lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU, names=[ "refined_anchors"])  # Tensor("ROI/refined_anchors:0", shape=(1, ?, 4), dtype=float32)

        # [batch, N, (y1, x1, y2, x2)]
        # 防止超出图片范围 (使用 utils.batch_slice 函数和 clip_boxes_graph 函数，对解码后的框进行裁剪，以防止超出图片范围，并将其存储在 boxes 变量中)
        '''
        对解码后的框进行裁剪，以防止超出图片范围:
            1. `window = np.array([0, 0, 1, 1], dtype=np.float32)`: 创建一个表示窗口的 NumPy 数组，
                         其中 `[0, 0, 1, 1]` 表示窗口的左上角坐标为 `(0, 0)`，右下角坐标为 `(1, 1)`。这个窗口通常用于定义图片的有效区域。
            2. `utils.batch_slice(boxes,...)`: 使用 `utils.batch_slice` 函数对 `boxes` 进行切片操作。
                - `boxes`: 要切片的张量，可能是解码后的框的坐标。  
                - `lambda x: clip_boxes_graph(x, window)`: 切片函数，它接受一个参数 `x`，并将其传递给 `clip_boxes_graph` 函数，同时传递窗口参数 `window`。
                                                          ->  裁剪边界框 clip-修剪  对边界框进行裁剪操作，确保边界框的坐标在指定的裁剪窗口范围内 
                - `self.config.IMAGES_PER_GPU`: 每个 GPU 要处理的图像数量。
                - `names=["refined_anchors_clipped"]`: 切片操作的名称。
            3. `clip_boxes_graph(x, window)`: 这是一个函数，用于对框进行裁剪。它接受框的坐标 `x` 和窗口参数 `window`，并返回裁剪后的框的坐标。
        综上所述，这段代码的作用是将解码后的框的坐标进行切片，并对每个切片应用裁剪函数，以确保框的坐标不会超出图片的有效区域。最后，将裁剪后的框的坐标存储在 `boxes` 变量中。这样可以保证后续的处理只考虑在图片范围内的框。
        '''
        window = np.array([0, 0, 1, 1], dtype=np.float32)  # [0. 0. 1. 1.]
        boxes = utils.batch_slice(boxes, lambda x: clip_boxes_graph(x, window), self.config.IMAGES_PER_GPU, names=[
            "refined_anchors_clipped"])  # Tensor("ROI/refined_anchors_clipped:0", shape=(1, ?, 4), dtype=float32)

        # 非极大抑制
        # 定义了一个非极大抑制的函数 nms，用于对框进行非极大抑制。然后，使用 utils.batch_slice 函数对框和得分进行切片，并应用非极大抑制函数，得到最终的建议框，并将其存储在 proposals 变量中
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(boxes,
                                                   scores,
                                                   self.proposal_count,
                                                   self.nms_threshold,
                                                   name="rpn_non_max_suppression")
            # 合并
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
# 其中，tf.log(x) 表示对 x 取自然对数，tf.log(2.0) 表示对 2 取自然对数, 作用是将输入的数值转换为以 2 为底的对数
def log2_graph(x):
    return tf.log(x) / tf.log(2.0)


# 获得输入进来的图像的大小
# image_meta: Tensor("input_image_meta:0", shape=(?, 93), dtype=float32)  1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES(81)
def parse_image_meta_graph(meta):
    """
    将meta里面的参数进行分割   meta[:, 0] -> 一维张量meta所有行的地一个元素(93个元素中的第一个)
    """
    # 1 -> 图像的唯一标识符  Tensor("roi_align_classifier/strided_slice:0", shape=(?,), dtype=float32)
    image_id = meta[:, 0]
    # 3 -> 原始图像的形状 Tensor("roi_align_classifier/strided_slice_1:0", shape=(?, 3), dtype=float32)
    original_image_shape = meta[:, 1:4]
    # 3 -> 当前图像的形状  Tensor("roi_align_classifier/strided_slice_2:0", shape=(?, 3), dtype=float32)
    image_shape = meta[:, 4:7]
    # 4 -> (y1, x1, y2, x2) window of image in in pixels
    #      图像在窗口中的位置（以像素为单位） Tensor("roi_align_classifier/strided_slice_3:0", shape=(?, 4), dtype=float32)
    window = meta[:, 7:11]
    # 1 -> 图像的缩放比例  Tensor("roi_align_classifier/strided_slice_4:0", shape=(?,), dtype=float32)
    scale = meta[:, 11]
    # self.NUM_CLASSES(81) -> 活动类别的标识符  Tensor("roi_align_classifier/strided_slice_5:0", shape=(?, 81), dtype=float32)
    active_class_ids = meta[:, 12:]
    # 将提取的参数以字典的形式返回
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


"""
定义 PyramidROIAlign 的类, 继承自 Layer 类
PyramidROIAlign 类用于实现金字塔 ROI 对齐操作，将不同大小的建议框映射到相应的特征层上，并进行截取和调整大小的操作。

    ROI 是 Region of Interest 的缩写，中文意思是“感兴趣区域”。在 Faster R-CNN 中，ROI 指的是图像中可能包含目标的区域, 这些区域通常是通过目标检测算法（如 Selective Search 、 Region Proposal Networks）或其他方法预先定义的。
    Feature Map 是卷积神经网络中卷积层的输出结果，它表示输入图像在不同卷积核下的特征响应。Feature Map 可以看作是输入图像的一种抽象表示，其中包含了图像的各种特征信息。
    ROI Pooling 是 Faster R-CNN 中的一个关键操作，将不同大小的 感兴趣区(ROI) 映射到 固定大小的特征图(Feature Map)上:                                                     
                1. **ROI 提取**：首先，需要从输入图像中提取出感兴趣的区域。这些区域可以是通过目标检测算法（如 Faster R-CNN）检测到的目标框，也可以是其他方式定义的区域。
                                    -. **目标检测算法**：使用目标检测算法（如 Faster R-CNN 本身或其他类似的算法）来检测输入图像中的目标。这些目标可以是各种物体、类别或感兴趣的区域。
                                    -. **目标框生成**：目标检测算法会输出一系列目标框，每个目标框表示检测到的目标的位置和大小。这些目标框通常以矩形的形式表示，具有左上角坐标和宽度、高度等尺寸信息。
                                    -. **ROI 提取**：根据目标框的位置和大小，从输入图像中提取出对应的感兴趣区域（ROI）。ROI 可以是目标框本身，也可以是根据需要进行一定的扩展或调整后的区域。
                2. **ROI 映射**：接下来，将提取到的 ROI 映射到特征图上。特征图是卷积神经网络在处理图像时生成的中间表示，它包含了图像的各种特征信息。
                                    -. 确定 ROI 的位置和大小：首先，需要确定提取到的 ROI 在输入图像中的位置和大小。
                                    -. 找到对应的特征图位置：根据 ROI 的位置和大小，在特征图中找到与之对应的位置。
                                    -. 映射 ROI 到特征图：将 ROI 中的像素值映射到特征图中对应的位置上。
                                    ROI Pooling 的目的是将不同大小的 ROI 转换为固定大小的特征表示，以便后续的处理和分类。通过池化操作，可以提取 ROI 中的主要特征，并减少尺寸差异对模型训练和推理的影响。
                                                将 ROI 划分为一个固定数量的子区域, 并对每个子区域进行最大池化操作, 得到固定大小的特征向量,
                                                    - 将输入的 ROI 区域划分为固定数量的子 区域,子区域的数量：通常根据具体的实现和需求来确定，常见的数量有 7x7 或 14x14 等
                                                    - 对每个子区域进行池化操作，通常采用最大池化
                                                    - 将池化后的结果组合成一个固定大小的特征向量, 一个固定的尺寸，通常与后续的全连接层或分类器的输入维度相匹配
                                                这个特征向量可以看作是 ROI 在特征图上的表示，它包含了 ROI 的各种特征信息。
                3. **特征图调整**：由于 ROI 的大小可能与特征图的大小不匹配，因此需要对特征图进行调整，使其能够与 ROI 进行对应。这可以通过插值、池化等操作来实现。                        
                                    - 可能的操作包括特征图的裁剪、缩放、填充等，以使其与特定的需求或后续处理步骤相匹配。
                4. **固定大小的特征图**：经过调整后，得到了一个与 ROI 大小相对应的固定大小的特征图。这个特征图可以作为后续处理的输入，例如用于目标分类、语义分割等任务。
                                      特征图输入到后续网络层
                                        - 将第 3 步得到的特征图作为输入，传递给后续的网络层。
                                        - 这些网络层可能包括全连接层、卷积层、分类器等，用于进行目标分类、边界框回归、掩码生成等任务。               
    通过 ROI Pooling，Faster R-CNN 可以处理不同大小的目标，并生成固定大小的特征图，以便后续的分类和回归任务。
    在得到特征向量后，通常会将其输入到全连接层或其他分类器中，进行目标的分类和回归。全连接层会将特征向量中的每个元素都连接到输出层的神经元上，从而实现对目标的分类和回归。          
"""
class PyramidROIAlign(Layer):

    # 初始化类的实例
    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    # 执行金字塔 ROI 对齐操作
    def call(self, inputs):
        """
            [<tf.Tensor 'ROI/packed_2:0' shape=(1, ?, 4) dtype=float32>,
             <tf.Tensor 'input_image_meta:0' shape=(?, 93) dtype=float32>,
             <tf.Tensor 'fpn_p2/BiasAdd:0' shape=(?, ?, ?, 256) dtype=float32>,
             <tf.Tensor 'fpn_p3/BiasAdd:0' shape=(?, ?, ?, 256) dtype=float32>,
             <tf.Tensor 'fpn_p4/BiasAdd:0' shape=(?, ?, ?, 256) dtype=float32>,
             <tf.Tensor 'fpn_p5/BiasAdd:0' shape=(?, ?, ?, 256) dtype=float32>]
        """
        # # # 1. 接收输入的参数 inputs，并将其分解为三个部分：boxes（建议框的位置）、image_meta（包含图片信息）和 feature_maps（所有的特征层）
        # 建议框的位置
        boxes = inputs[0]  # Tensor("ROI/packed_2:0", shape=(1, ?, 4), dtype=float32)
        # image_meta包含了一些必要的图片信息
        image_meta = inputs[1]  # Tensor("input_image_meta:0", shape=(?, 93), dtype=float32)
        # 取出所有的特征层[batch, height, width, channels]
        feature_maps = inputs[2:]
        # 通过 tf.split 函数将 boxes 按照维度 2(第3个维度) 拆分成四个部分，分别赋值给变量 y1、x1、y2 和 x2。然后计算建议框的高度 h 和宽度 w
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # 获得输入进来的图像的大小
        # image_meta: Tensor("input_image_meta:0", shape=(?, 93), dtype=float32)  1+3+3+4+1+self.NUM_CLASSES(81)
        # Tensor("roi_align_classifier/strided_slice_6:0", shape=(3,), dtype=float32)
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]

        # # # # 2. 通过建议框的大小找到这个建议框属于哪个特征层
        '''
        tf.cast() 用于将张量的数据类型转换为指定的数据类型
        tf.minimum() 用于计算两个张量的元素级最小值, 返回一个新的张量，其元素值是两个输入张量对应位置元素的最小值
           例： 创建两个张量 tensor1 = tf.constant([1, 2, 3]) tensor2 = tf.constant([4, 5, 6])
                          result = tf.minimum(tensor1, tensor2) -> [1, 2, 3] 
        tf.maximum() 用于计算两个张量的元素级最大值, 其元素值是两个输入张量对应位置元素的最大值
        tf.sqrt() 用于计算张量的平方根。它接受一个张量作为输入，并返回一个与输入张量形状相同的张量，其中每个元素都是输入张量对应元素的平方根。
           例： tensor = tf.constant([4, 9, 16]) -> tf.sqrt(tensor) = [2. 3.  4. ]
        tf.squeeze() 用于删除张量中维度为 2 的维度, 即第3维度。如果原始张量中没有维度为2的维度，则 tf.squeeze() 函数不会执行任何操作
        '''
        # 计算建议框的面积 image_area  Tensor("roi_align_classifier/mul:0", shape=(), dtype=float32)
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # [建议框所属的特征层的级别]
        # 通过计算建议框的平方根与图像面积的平方根的比值，并取对数，得到建议框所属的特征层的级别。
        # Tensor("roi_align_classifier/truediv_3:0", shape=(1, ?, 1), dtype=float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        # 通过 tf.minimum 和 tf.maximum 函数对 roi_level 进行限制，确保其在有效范围内, 将 roi_level 限制在 2 到 5 之间
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        # 压缩 roi_level 的维度：使用 tf.squeeze 函数将 roi_level 的维度从 (1,?, 1) 压缩为 (1,?)
        # roi_level 张量的形状将变为 (batch_size, num_boxes) Tensor("roi_align_classifier/Squeeze:0", shape=(1, ?), dtype=int32)
        roi_level = tf.squeeze(roi_level, 2)

        # # # 3. 通过循环遍历特征层，对于每个特征层，找到对应的建议框，并使用 tf.image.crop_and_resize 函数进行截取和调整大小。将截取后的结果添加到列表 pooled 中
        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        # 分别在P2-P5中进行截取
        '''
        `enumerate()` 是 Python 内置的一个函数，用于将一个可迭代对象转换为包含索引和元素的元组序列。
                my_list = ['apple', 'banana', 'cherry']
                # 使用 enumerate() 函数
                for index, element in enumerate(my_list):
                    print(f'索引 {index}: {element}')
            在上述示例中，`enumerate(my_list)` 会返回一个迭代器，该迭代器会生成一系列的元组，其中每个元组的第一个元素是索引，第二个元素是对应的元素。
            通过使用 `for` 循环遍历这个迭代器，我们可以方便地获取每个元素的索引和值，并进行相应的处理。
        `enumerate()` 函数常用于需要同时访问元素和其索引的情况，例如在遍历列表、字符串、元组等可迭代对象时。
        '''
        for i, level in enumerate(range(2, 6)):

            # 找到每个特征层对应box  Tensor("roi_align_classifier/Where:0", shape=(?, 2), dtype=int64)  2-满足条件的行索引、列索引
            '''
            ix 是一个张量，它表示在 roi_level 张量中 等于 给定特征层 level(P2) 的元素的索引
            使用 TensorFlow 的 `tf.where()` 函数来查找满足条件的元素索引:
                具体来说，将 roi_level 张量中的每个元素与给定的特征层级别 level(P2) 进行比较
                        返回一个布尔类型的张量，其中 `True` 表示 `roi_level` 和 `level` 相等的位置，`False` 表示不相等的位置。
                然后，`tf.where()` 函数会返回一个元组，其中第一个元素是满足条件的行索引，第二个元素是满足条件的列索引。
            在这个例子中，`ix` 将会是一个张量，其中包含了满足条件的元素的索引。
            '''
            ix = tf.where(tf.equal(roi_level, level))  # level -> 2
            '''
            [level_boxes-给定特征层(P2)相关的建议框]   Tensor("roi_align_classifier/GatherNd:0", shape=(?, 4), dtype=float32)
            tf.gather() 和 tf.gather_nd() 都是 TensorFlow 中的函数，用于从张量中收集元素。
            tf.gather() 函数用于根据给定的索引从张量中收集元素。它接受一个张量 params 和一个索引张量 indices，并返回一个新的张量，其中包含根据索引从 params 中收集的元素。
                        params = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  
                        indices = tf.constant([0,2])
                        gathered = tf.gather_nd(params, indices)  -> [[1 2 3][7 8 9]]
            tf.gather_nd() 函数根据给定的多维索引从输入张量中收集元素。索引是一个多维张量，其中每一行表示一个元素的索引。其中包含根据索引从 params 中收集的元素。
                        params = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                        indices = tf.constant([[0, 0], [1, 2]])
                        gathered = tf.gather_nd(params, indices)  -> [1 6]
                        
            执行 tf.gather_nd(boxes, ix) 后，将返回一个新的张量 level_boxes，其中包含了根据索引 ix 从 boxes 中选取的子张量, 这些子张量对应于与给定特征层相关的建议框。
            使用 TensorFlow 的 `gather_nd` 函数从 `boxes` 张量中根据索引 `ix` 选取子张量:                
                - `boxes`：这是一个张量，可能表示一个多维数组或张量。
                - `ix`：这是一个索引张量，它指定了要从 `boxes` 中选取的子张量的位置。
            通过执行 `tf.gather_nd(boxes, ix)`，代码将根据索引 `ix` 从 `boxes` 中选取相应位置的子张量，并将结果存储在 `level_boxes` 变量中。                
            '''
            level_boxes = tf.gather_nd(boxes, ix)  # Tensor("roi_align_classifier/GatherNd:0", shape=(?, 4), dtype=float32)
            '''
            [box_to_level - 给定特征层 level(P2) 的元素的索引]    [N, M]，其中 N 表示建议框的数量，M 表示每个建议框的索引信息
            '''
            box_to_level.append(ix)  # {list:1} [<tf.Tensor 'roi_align_classifier/Where:0' shape=(?, 2) dtype=int64>] 2-满足条件的 ()

            # 获得这些box所属的图片
            '''
            [box_indices - 每个元素表示对应的框所属的图片索引]
            将张量 `ix` 的第一列转换为 `tf.int32` 类型，并将结果存储在变量 `box_indices` 中。                
                - `ix` 是一个张量，它的第一列表示每个框所属的图片索引。
                - `tf.cast(ix[:, 0], tf.int32)` 使用 TensorFlow 的 `cast` 函数将 `ix` 的第一列转换为 `tf.int32` 类型。
                - `box_indices` 是转换后的结果，它是一个整数类型的张量，每个元素表示对应的框所属的图片索引。
            通过将 `ix` 的第一列转换为整数类型，可以方便地根据图片索引对框进行操作或处理。                
            '''
            box_indices = tf.cast(ix[:, 0], tf.int32)  # Tensor("roi_align_classifier/Cast_1:0", shape=(?,), dtype=int32)

            # 停止梯度下降
            '''
            使用 TensorFlow 的 `tf.stop_gradient()` 函数来停止计算张量的梯度:
                在深度学习中，梯度用于更新模型的参数。然而，在某些情况下，我们可能不希望某些张量的梯度被计算或传播。这可以用于固定某些参数的值，或者在训练过程中避免某些中间结果的梯度对模型的影响。
                在这段代码中，`level_boxes` 和 `box_indices` 是两个张量。通过将它们传递给 `tf.stop_gradient()` 函数，意味着在后续的计算中，不会对这两个张量计算梯度。
                这样做的目的可能是为了在训练过程中固定这些张量的值，或者在计算过程中避免它们对梯度的影响。
                需要注意的是，停止梯度计算可能会影响到模型的训练和优化过程。在使用 `tf.stop_gradient()` 函数时，需要谨慎考虑其对模型性能和训练效果的影响，并确保这样做是符合你的需求的。
            '''
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            '''
            `tf.image.crop_and_resize()` 是 TensorFlow 中的一个函数，用于对图像进行裁剪和调整大小操作:
                1. `image`：要进行裁剪和调整大小的输入图像张量。
                2. `boxes`：裁剪框的坐标张量。它应该是一个形状为 `[num_boxes, 4]` 的二维张量，其中每个行表示一个裁剪框的左上角和右下角坐标。
                3. `box_ind`：每个裁剪框在输入图像中的索引张量。它应该是一个形状为 `[num_boxes]` 的一维张量。
                4. `crop_size`：裁剪后的目标大小。它可以是一个整数，表示裁剪后的正方形大小，也可以是一个形状为 `[height, width]` 的二维张量，表示裁剪后的矩形大小。
                5. `method`：插值方法，用于在调整大小时进行像素值的插值。常见的插值方法包括 "bilinear"（双线性插值）、"nearest"（最近邻插值）等。
                6. `extrapolation_value`：在裁剪框外的像素值。当裁剪框超出图像边界时，该值将用于填充超出部分的像素。
            该函数的作用是根据提供的裁剪框和目标大小，对输入图像进行裁剪，并使用指定的插值方法将裁剪后的图像调整大小到目标大小。
            返回的结果是一个与裁剪框数量相同的张量，每个张量表示一个裁剪和调整大小后的图像。
                        
            对特征图进行裁剪和调整大小的操作，并将结果添加到 `pooled` 列表中。                
                - `feature_maps[i]` 是(P2-P5中的) 第 `i` 个特征图。
                - `level_boxes` 是对应特征层的建议框。
                - `box_indices` 是建议框在特征图中的索引。
                - `self.pool_shape` 是池化操作的形状。
                - `method="bilinear"` 指定了使用双线性插值的方法进行调整大小。
                通过 `tf.image.crop_and_resize()` 函数，根据建议框的位置和索引，从特征图中裁剪出相应的区域，并将其调整大小到指定的池化形状。
                最后，将调整大小后的结果添加到 `pooled` 列表中。
            这样，在循环结束后，`pooled` 列表将包含所有特征层的裁剪和调整大小后的结果。       
            pooled ->  {list:4} Tensor("roi_align_classifier/CropAndResize:0", shape=(?, 7, 7, 256), dtype=float32) ...         
            '''
            pooled.append(tf.image.crop_and_resize(feature_maps[i], level_boxes, box_indices, self.pool_shape, method="bilinear"))

        # # # 4. 将 pooled 列表中的结果进行拼接，并根据建议框的顺序进行排序，将同一张图里的建议框聚集在一起。
        '''
        将 `pooled` 列表中的所有张量沿着指定的轴（在这里是 `axis=0`）进行拼接:
            `pooled` 列表中包含了多个裁剪和调整大小后的特征图，通过将它们沿着 `axis=0` 进行拼接，可以得到一个形状为 `[batch * num_boxes, pool_height, pool_width, channels]` 的张量。
        这样的拼接操作在深度学习中常用于 将多个样本或多个特征图组合在一起 ,以便进行后续的处理或计算。
        '''
        pooled = tf.concat(pooled, axis=0)  # Tensor("roi_align_classifier/concat:0", shape=(?, 7, 7, 256), dtype=float32)

        # # 将顺序和所属的图片进行堆叠
        # 将 box_to_level 进行堆叠和扩展，然后根据索引进行排序，将同一张图里的建议框聚集在一起。最后，根据排序后的索引获取图片的索引，并从 pooled 中选择相应的部分
        '''
        [box_to_level - 所有建议框的索引信息]concat_1
        所有建议框的索引信息: 将 box_to_level 列表中的所有张量沿着指定的轴（在这里是 axis=0）进行拼接
        具体来说，box_to_level 列表中可能包含了多个与建议框对应的索引张量。通过使用 tf.concat() 函数将这些张量拼接在一起，可以得到一个更大的张量，其中包含了所有建议框的索引信息。
        '''
        # Tensor("roi_align_classifier/:0", shape=(?, 2), dtype=int64)
        box_to_level = tf.concat(box_to_level, axis=0)
        ''' 
         box_range -> 新的张量 `box_range`代表扩展之后的box to level的shape  其形状为 `[N, 1]` 
        `box_range` 是通过在 `box_to_level` 张量的第一维上扩展一个维度得到的张量:
            `box_to_level` 张量的形状为 `[N, M]`，其中 `N` 表示建议框的数量，`M` 表示每个建议框的索引信息。   
            tf.range(tf.shape(box_to_level)[0])` 创建了一个从 0 到 `tf.shape(box_to_level)[0] - 1` 的整数序列张量。
            然后，`tf.expand_dims()` 函数将这个序列张量的第一维扩展为 1，得到一个形状为 [tf.shape(box_to_level)[0], 1] 的张量 box_range。         
            通过执行 `tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)`，创建了一个新的张量 `box_range`，其形状为 `[N, 1]`。           
            在这个张量中，每一行都包含一个从 0 到 `N-1` 的整数序列,`box_range` 的作用通常是为了在后续的计算中，与其他张量进行组合或对齐，以便对建议框进行更复杂的操作。                                  
        '''
        # Tensor("roi_align_classifier/ExpandDims:0", shape=(?, 1), dtype=int32)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        '''
        [box_to_level -> 每个建议框的唯一标识符或索引]
        将 `box_to_level` 和 `box_range` 这两个张量沿着第二个维度（`axis=1`）进行连接:
            首先，`tf.cast(box_to_level, tf.int32)` 将 `box_to_level` 张量的数据类型转换为整数类型 `tf.int32`。
            然后，通过 `tf.concat()` 函数将转换后的 `box_to_level` 和 `box_range` 张量连接在一起。连接后的结果将是一个新的张量，其维度与连接前的两个张量的维度之和相同。
            将 `box_range` 与 `box_to_level` 张量进行连接，以获取每个建议框的唯一标识符或索引。      
        '''
        # Tensor("roi_align_classifier/concat_2:0", shape=(?, 3), dtype=int32)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)
        '''
        创建了一个排序张量 `sorting_tensor`，它是通过将 `box_to_level` 张量的第一列和第二列的值相乘，并将结果相加得到的:
            box_to_level[:, 0]` 表示 `box_to_level` 张量的第一列，`表示第几张图, 即 建议框所属的图像索引
            box_to_level[:, 1]` 表示 `box_to_level` 张量的第二列， 表示表示第几张图里的第几个框， 即 建议框在图像中的索引

            代码 `sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]` 中，乘以 `100000` 的目的是为了增加第一列的值的权重，以便在排序时更强调第一列的差异 
                通过乘以一个较大的数，可以使第一列的值在排序中占据更大的比重，从而对排序结果产生更大的影响。这样做可以根据具体的需求来调整排序的优先级。
                                    即使图像索引的差异在排序中更加突出，从而首先按照图像索引进行排序，然后在每个图像内部按照建议框的索引进行排序。
                通过将这两列的值相乘，并将结果相加，得到了一个新的张量 `sorting_tensor`。这个张量的值将用于对 `box_to_level` 张量进行排序。
            在后续的代码中，通常会使用 `sorting_tensor` 来对 `box_to_level` 张量进行排序，以便按照以上特定的顺序处理建议框。
        '''
        # Tensor("roi_align_classifier/add_1:0", shape=(?,), dtype=int32)
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        # 进行排序，将同一张图里的某一些聚集在一起
        '''
        `tf.nn.top_k()` 函数来获取排序张量 `sorting_tensor` 中前 `k` 个最大值的索引:
            返回一个包含两个元素的元组, 第一个元素是值张量，其中包含了排序张量中的前 `k` 个最大值。第二个元素是索引张量，其中包含了这些最大值在原始排序张量中的索引。
                    
            `k` 的值被设置为 `tf.shape(box_to_level)[0]`，这意味着将返回排序张量中所有值的索引。
            indices[::-1] 是 Python 中的切片操作，用于反转索引张量 indices 的顺序，这样就得到了按照降序排列的索引
                   切片操作可以通过指定起始索引、结束索引和步长来提取张量或列表的一部分。[::-1] 表示从末尾开始，以步长为-1 的方式提取整个张量或列表。
        '''
        # Tensor("roi_align_classifier/strided_slice_17:0", shape=(?,), dtype=int32)
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        # 按顺序获得图片的索引
        '''
        box_to_level 是一个张量，它的形状为 [N, 3]，其中 N 表示建议框的数量，3 表示每个建议框的索引信息。
        box_to_level[:, 2] 表示取 box_to_level 张量的第三列，即每个建议框在特征图中的索引。 (代表取出box to level的所有维度的shape)
        根据索引 `ix` 从 `box_to_level` 张量的第三列中选择相应的元素:
            `tf.gather()` 函数的作用是根据提供的索引从输入张量中选择特定位置的元素。
                         `box_to_level[:, 2]` 表示 `box_to_level` 张量的第三列，`ix` 是一个索引张量，它指定了要选择的元素的位置。
            这样做的目的可能是为了根据前面计算得到的索引 `ix`，从 `box_to_level` 张量中选择与排序后的建议框对应的图片索引。            
        '''
        # Tensor("roi_align_classifier/GatherV2:0", shape=(?,), dtype=int32)
        ix = tf.gather(box_to_level[:, 2], ix)
        '''
        使用 TensorFlow 的 `tf.gather()` 函数根据索引 `ix` 从 `pooled` 张量中选择相应的元素: 
            `tf.gather()` 函数的作用是根据提供的索引从输入张量中选择特定位置的元素。 根据前面计算得到的索引 `ix`，从 `pooled` 张量中选择与排序后的建议框对应的部分。            
        '''
        # Tensor("roi_align_classifier/GatherV2_1:0", shape=(?, 7, 7, 256), dtype=float32)
        pooled = tf.gather(pooled, ix)

        # # 重新reshape为原来的格式 将截取和调整大小后的结果重新调整为原始的格式，返回最终的输出
        # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
        '''
        将 `boxes` 张量的前两维形状和 `pooled` 张量的后三维形状连接在一起，得到一个新的形状张量 `shape`:
            具体来说，`tf.shape(boxes)[:2]` 表示 `boxes` 张量的前两维形状，即批量大小和建议框的数量。
                     `tf.shape(pooled)[1:]` 表示 `pooled` 张量的后三维形状，即池化后的高度、宽度和通道数。
            通过使用 `tf.concat()` 函数将这两个形状张量连接在一起，得到一个新的形状张量 `shape`，其中包含了批量大小、建议框的数量、池化后的高度、宽度和通道数。
        '''
        # Tensor("roi_align_classifier/concat_3:0", shape=(5,), dtype=int32)
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        #  pooled -> Tensor("roi_align_classifier/Reshape:0", shape=(1, ?, 7, 7, 256), dtype=float32)
        pooled = tf.reshape(pooled, shape)
        # Tensor("roi_align_classifier/Reshape:0", shape=(1, ?, 7, 7, 256), dtype=float32)
        #        [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


# --------------------------------------------------------------------------------------#
#   Detection Layer  refine-精练
#   对分类建议进行细化和过滤，去除重叠的部分，并返回最终的检测结果。检测结果包括建议框的坐标、类 ID 和得分
# --------------------------------------------------------------------------------------#
def refine_detections_graph(rois, probs, deltas, window, config):
    """细化分类建议并过滤重叠部分并返回最终结果探测。
    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image that contains the image excluding the padding.
    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where coordinates are normalized.
    """

    # 找到得分最高的类
    '''
     找到每个建议框中得分最高的类的 ID:
         `tf.argmax` 是 TensorFlow 中的一个函数，用于在张量中找到最大值的索引。
            tf.argmax(input, axis=None, output_type=tf.int64, name=None)
                - `input`：要进行操作的张量。
                - `axis`：指定要在哪个轴上进行操作。如果不指定，则在整个张量中进行操作。
                - `output_type`：指定输出的类型，默认为 `tf.int64`。
                - `name`：操作的名称（可选）。
        `tf.argmax` 函数返回一个张量，其中包含输入张量中最大值的索引。
             input =  [[0, 1, 2], 
                       [3, 4, 5]] , 
            `tf.argmax(input, axis=0)` 将返回 `[1, 1, 1]`，表示在第一维上最大值的索引为 1。
            `tf.argmax(input, axis=1)` 将返回 `[2, 2]`，表示在第二维上最大值的索引为 2。
        如果需要获取最大值本身，可以使用 `tf.reduce_max` 函数。
    '''
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)  # Tensor("mrcnn_detection/ArgMax:0", shape=(1000,), dtype=int32)

    # 序号+类
    '''
    将建议框的索引和对应的类 ID 组合成一个索引张量:
        - `tf.range(probs.shape[0])` 生成一个从 0 到 `probs.shape[0] - 1` 的整数序列张量。
        - `class_ids` 是一个表示每个建议框所属类别的张量。
        - `tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)` 将这两个张量沿着新的维度（第二维，`axis=1`）进行堆叠。
                    将 `tf.range(probs.shape[0])` 和 `class_ids` 看作是一行一行的元素，然后将它们堆叠在一起形成一个新的张量。
        最终得到的 `indices` 张量的形状为 `(probs.shape[0], 2)`，其中第一列是建议框的索引，第二列是对应的类 ID。        
    '''
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)  # Tensor("mrcnn_detection/stack:0", shape=(1000, 2), dtype=int32)

    # 取出成绩
    '''
     根据索引张量取出每个建议框中对应类的得分:
        使用 TensorFlow 的 `tf.gather_nd()` 函数根据索引 `indices` 从概率张量 `probs` 中收集对应的类得分。
            - `probs` 是一个张量，表示每个建议框属于每个类的概率。它的形状可能是 `[N, num_classes]`，其中 `N` 是建议框的数量，`num_classes` 是类的数量。
            - `indices` 是一个张量，它的形状是 `[M, 2]`，其中 `M` 是要收集的元素数量。`indices` 的每一行表示一个要收集的元素的索引，其中第一列是建议框的索引，第二列是类的索引。
            - `tf.gather_nd(probs, indices)` 会根据 `indices` 中的索引从 `probs` 中收集对应的类得分。它返回一个张量，其形状是 `[M]`，其中每个元素是对应建议框和类的得分。
        通过这种方式，可以根据指定的建议框和类的索引从概率张量中提取出对应的类得分，以便进行后续的处理或分析。
    '''
    class_scores = tf.gather_nd(probs, indices)  # Tensor("mrcnn_detection/GatherNd:0", shape=(1000,), dtype=float32)

    # 还有框的调整参数
    '''
    根据索引张量取出每个建议框中对应类的边界框调整参数:
        使用 TensorFlow 的 `tf.gather_nd()` 函数根据索引 `indices` 从张量 `deltas` 中收集特定的元素。
            1. `deltas` 是一个张量，它可能包含了与不同建议框和类别相关的一些数据。
            2. `indices` 是一个索引张量，它指定了要从 `deltas` 中收集的元素的位置。
            3. `tf.gather_nd(deltas, indices)` 会根据 `indices` 中的索引值，从 `deltas` 中收集对应的元素，并将它们组合成一个新的张量 `deltas_specific`。
    这样，`deltas_specific` 就包含了根据特定索引选择的 `deltas` 中的元素。
    '''
    deltas_specific = tf.gather_nd(deltas, indices)  # Tensor("mrcnn_detection/GatherNd_1:0", shape=(1000, 4), dtype=float32)

    # 进行解码
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    '''
    使用边界框调整参数对建议框进行调整，得到细化后的边界框:        
        1. `rois`：这是一个张量，表示建议框的坐标。它的形状可能是 `[N, 4]`，其中 `N` 是建议框的数量，每个建议框由四个坐标值（例如左上角和右下角的坐标）表示。  [N, (y1, x1, y2, x2)]
        2. `deltas_specific`：这是一个张量，它包含了与每个建议框和类别相关的边界框调整参数。它的形状可能是 `[N, num_classes, 4]`，其中 `num_classes` 是类别的数量。 [N, num_classes, (dy, dx, log(dh), log(dw))]
        3. `config.BBOX_STD_DEV`：这是一个配置参数，可能表示边界框调整参数的标准差。  [0.1, 0.1, 0.2, 0.2]
        4. `deltas_specific * config.BBOX_STD_DEV`：这是对边界框调整参数进行缩放，根据标准差进行调整。
        5. `apply_box_deltas_graph(rois, deltas_specific * config.BBOX_STD_DEV)`：将边界框的调整量应用到先验框上的功能，得到了调整后的边界框坐标
        6. `refined_rois`：这是调整后的边界框张量，它的形状与 `rois` 相同，但包含了更精确的边界框坐标。
    通过这种方式，可以根据建议框和对应的边界框调整参数来计算更精确的边界框，这在目标检测等任务中常用于对检测结果进行微调。
    '''
    refined_rois = apply_box_deltas_graph(rois, deltas_specific * config.BBOX_STD_DEV)  # Tensor("mrcnn_detection/apply_box_deltas_out:0", shape=(1000, 4), dtype=float32)

    # 防止超出0-1
    '''将细化后的边界框裁剪到窗口范围内，确保边界框不超出图像范围。'''
    refined_rois = clip_boxes_graph(refined_rois, window)  # Tensor("mrcnn_detection/clipped_boxes:0", shape=(1000, 4), dtype=float32)

    # 去除背景
    '''
    去除背景类（类 ID 为 0）的建议框:
        使用 TensorFlow 的 `tf.where()` 函数来找到类 ID 大于 0 的建议框的索引，并将这些索引存储在变量 `keep` 中:
            1. `class_ids` 是一个张量，表示每个建议框所属的类 ID。(每个建议框中得分最高的类的 ID)
            2. `tf.where(class_ids > 0)` 会返回一个元组，其中第一个元素是满足条件（类 ID 大于 0）的建议框的索引，第二个元素是一个空张量。
            3. `[:, 0]` 用于从返回的元组中选择第一个元素，即满足条件的建议框的索引。
            4. 最终，`keep` 张量将包含类 ID 大于 0 的建议框的索引。
        这样，`keep` 张量就可以用于后续的操作，例如保留这些建议框进行进一步的处理或分析。
    '''
    keep = tf.where(class_ids > 0)[:, 0]  # Tensor("mrcnn_detection/strided_slice_22:0", shape=(?,), dtype=int64)

    # 去除背景和得分小的区域
    '''
    根据配置中的最小置信度阈值，进一步筛选建议框。 
        1. `config.DETECTION_MIN_CONFIDENCE`：这是一个配置参数，表示最小置信度阈值。
        2. `tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]`：使用 `tf.where()` 函数找到类得分大于或等于最小置信度阈值的建议框的索引。   
        3. `tf.expand_dims(keep, 0)` 和 `tf.expand_dims(conf_keep, 0)`： 使用 `tf.expand_dims()` 函数将 `keep` 和 `conf_keep` 张量扩展为二维张量，以便进行集合操作。
        4. `tf.sets.set_intersection()`：这是 TensorFlow 中的集合操作函数，用于计算两个集合的交集。在这里，它用于计算 `keep` 和 `conf_keep` 的交集。
        5. `tf.sparse_tensor_to_dense(keep)[0]`：将交集结果转换为稠密张量，并通过 `[0]` 索引获取第一维的元素，即最终保留的建议框索引。
    综上所述，这段代码的目的是根据最小置信度阈值进一步筛选建议框，只保留类得分大于或等于该阈值的建议框。这样可以提高检测结果的准确性，减少误检。
    '''
    if config.DETECTION_MIN_CONFIDENCE:  # 0.7  如果设置了最小置信度阈值，则进一步去除得分低于阈值的建议框
        # Tensor("mrcnn_detection/strided_slice_23:0", shape=(?,), dtype=int64)
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        # SparseTensor(indices=Tensor("mrcnn_detection/DenseToDenseSetOperation:0", shape=(?, 2), dtype=int64),
        #              values=Tensor("mrcnn_detection/DenseToDenseSetOperation:1", shape=(?,), dtype=int64),
        #              dense_shape=Tensor("mrcnn_detection/DenseToDenseSetOperation:2", shape=(2,), dtype=int64))
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
        # Tensor("mrcnn_detection/strided_slice_24:0", shape=(?,), dtype=int64)
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # 获得除去背景并且得分较高的框还有种类与得分
    # 1. Prepare variables
    # 收集保留下来的建议框的类 ID        Tensor("mrcnn_detection/GatherV2:0", shape=(?,), dtype=int32)
    pre_nms_class_ids = tf.gather(class_ids, keep)
    # 收集保留下来的建议框的得分         Tensor("mrcnn_detection/GatherV2_1:0", shape=(?,), dtype=float32)
    pre_nms_scores = tf.gather(class_scores, keep)
    # 收集保留下来的建议框的边界框坐标    Tensor("mrcnn_detection/GatherV2_2:0", shape=(?, 4), dtype=float32)
    pre_nms_rois = tf.gather(refined_rois, keep)
    #  找到保留下来的建议框中唯一的类 ID  Tensor("mrcnn_detection/Unique:0", shape=(?,), dtype=int32)
    '''
    `tf.unique()` 是 TensorFlow 中的一个函数，用于查找张量中的唯一元素: 
        tf.unique(input, out_idx=tf.int32, name=None)    
            - `input`：要查找唯一元素的张量。
            - `out_idx`：可选参数，指定输出的索引张量的类型。默认值为 `tf.int32`。
        `tf.unique()` 函数返回两个张量：
            - `output`：包含输入张量中的唯一元素。
            - `idx`：包含每个唯一元素在输入张量中的原始索引。
        例如，如果你有一个张量 `t` ，其中包含一些重复的元素，你可以使用 `tf.unique()` 函数来查找唯一元素，并获取它们的原始索引：
            t = tf.constant([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
            output, idx = tf.unique(t)
            print(output.numpy())   -> [1, 2, 3, 4]
            print(idx.numpy())      -> [0 1 1 2 2 2 3 3 3 3]   1 在 t 中的索引是 0，2 在 t 中的索引是 1 和 2，3 在 t 中的索引是 3、4 和 5 
    '''
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    # 定义一个函数，用于对每个类进行非极大抑制（NMS）操作
    def nms_keep_map(class_id):
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]

        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)

        # Tensor("mrcnn_detection/map/while/GatherV2_3:0", shape=(?,), dtype=int64)
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))

        # Tensor("mrcnn_detection/map/while/sub:0", shape=(), dtype=int32)
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        # Tensor("mrcnn_detection/map/while/PadV2:0", shape=(?,), dtype=int64)
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)

        # <bound method Tensor.set_shape of <tf.Tensor 'mrcnn_detection/map/while/PadV2:0' shape=(100,) dtype=int64>>
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep  # Tensor("mrcnn_detection/map/while/PadV2:0", shape=(100,), dtype=int64)

    # 2. 进行非极大抑制
    '''
     对每个类应用 NMS 操作，得到每个类的保留建议框的索引:
         使用 TensorFlow 的 `tf.map_fn()` 函数对 `unique_pre_nms_class_ids` 中的每个元素应用 `nms_keep_map` 函数，并将结果存储在 `nms_keep` 张量中。
            - `nms_keep_map` 是一个函数，它接受一个类 ID 作为输入，并返回一个布尔值，表示该类是否应该被保留。
            - `unique_pre_nms_class_ids` 是一个张量，其中包含了唯一的类 ID。
            - `tf.map_fn()` 函数将 `nms_keep_map` 函数应用于 `unique_pre_nms_class_ids` 中的每个元素，并将结果存储在 `nms_keep` 张量中。`dtype=tf.int64` 指定了 `nms_keep` 张量的数据类型。
     通过这种方式，可以对每个类 ID 应用 `nms_keep_map` 函数，并得到一个布尔值张量，表示每个类是否应该被保留。这在非极大值抑制（NMS）等操作中常用于确定哪些检测框应该被保留。
    '''
    # Tensor("mrcnn_detection/map/TensorArrayStack/TensorArrayGatherV3:0", shape=(?, 100), dtype=int64)
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)

    # 3. 找到符合要求的需要被保留的建议框 将 NMS 操作得到的索引张量重塑为一维张量
    nms_keep = tf.reshape(nms_keep, [-1])  # Tensor("mrcnn_detection/Reshape:0", shape=(?,), dtype=int64)
    '''
    去除 NMS 操作中被抑制的建议框的索引:
        使用 TensorFlow 的 `tf.gather()` 函数从 `nms_keep` 张量中选择满足条件的元素
        - `tf.where(nms_keep > -1)[:, 0]` 找到 `nms_keep` 张量中大于 `-1` 的元素的索引。
        - `tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])` 使用这些索引从 `nms_keep` 张量中选择对应的元素。
    '''
    # Tensor("mrcnn_detection/GatherV2_3:0", shape=(?,), dtype=int64)
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])

    # 4. Compute intersection between keep and nms_keep
    '''
    计算保留下来的建议框和 NMS 操作保留下来的建议框的交集:
        使用 TensorFlow 的 `tf.sets.set_intersection()` 函数计算两个集合的交集        
            - `tf.expand_dims(keep, 0)` 将 `keep` 张量扩展为一个二维张量，增加了一个维度。
            - `tf.expand_dims(nms_keep, 0)` 将 `nms_keep` 张量也扩展为一个二维张量，增加了一个维度。
            - `tf.sets.set_intersection()` 函数接受两个二维张量作为输入，并计算它们的交集。
        通过这种方式，可以计算 `keep` 和 `nms_keep` 这两个集合的交集，并将结果存储在 `keep` 张量中。
    '''
    # SparseTensor(indices=Tensor("mrcnn_detection/DenseToDenseSetOperation_1:0", shape=(?, 2), dtype=int64),
    #              values=Tensor("mrcnn_detection/DenseToDenseSetOperation_1:1", shape=(?,), dtype=int64),
    #              dense_shape=Tensor("mrcnn_detection/DenseToDenseSetOperation_1:2", shape=(2,), dtype=int64))
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))

    '''
    将交集转换为稠密张量：
       将稀疏张量 `keep` 转换为稠密张量，并获取转换后的稠密张量的第一维元素。
            1. `tf.sparse_tensor_to_dense(keep)`：这是 TensorFlow 中的一个函数，用于将稀疏张量转换为稠密张量。稀疏张量通常用于表示稀疏数据，其中只有一部分元素具有非零值。
            2. `[0]`：这是对转换后的稠密张量进行索引操作。通过 `[0]`，我们获取稠密张量的第一维元素。
    综上所述，这行代码的效果是将稀疏张量 `keep` 转换为稠密张量，并获取该稠密张量的第一维元素。这样可以方便后续的处理和操作，特别是当需要对稠密张量进行进一步的计算或与其他张量进行交互时。
    '''
    # Tensor("mrcnn_detection/strided_slice_26:0", shape=(?,), dtype=int64)
    keep = tf.sparse_tensor_to_dense(keep)[0]

    # 寻找得分最高的num_keep个框
    roi_count = config.DETECTION_MAX_INSTANCES  # 设置要保留的建议框数量  100
    # 收集保留下来的建议框的得分  Tensor("mrcnn_detection/GatherV2_4:0", shape=(?,), dtype=float32)
    class_scores_keep = tf.gather(class_scores, keep)
    # 根据得分和要保留的建议框数量，确定实际要保留的建议框数量  Tensor("mrcnn_detection/Minimum_4:0", shape=(), dtype=int32)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    # 找到得分最高的前 num_keep 个建议框的索引  Tensor("mrcnn_detection/TopKV2:1", shape=(?,), dtype=int32)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    # 根据索引保留得分最高的建议框  Tensor("mrcnn_detection/GatherV2_5:0", shape=(?,), dtype=int64)
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # 将保留下来的建议框的坐标、类 ID 和得分组合成一个检测结果张量  Tensor("mrcnn_detection/concat_1:0", shape=(?, 6), dtype=float32)
    '''
    使用 TensorFlow 的 `tf.concat()` 函数将多个张量沿着指定的轴进行连接，形成一个新的张量 `detections`。
        1. `tf.gather(refined_rois, keep)`：根据索引张量 `keep` 从 `refined_rois` 张量中选择对应的元素。
        2. `tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis]`：将索引张量 `keep` 从 `class_ids` 张量中选择的元素转换为浮点数，并在最后添加一个新的维度。
        3. `tf.gather(class_scores, keep)[..., tf.newaxis]`：根据索引张量 `keep` 从 `class_scores` 张量中选择对应的元素，并在最后添加一个新的维度。
        4. `tf.concat([...], axis=1)`：将上述三个张量沿着第二维（`axis=1`）进行连接。
    最终得到的 `detections` 张量将包含建议框的坐标、类 ID 和得分，这些信息将用于后续的处理或分析。
    '''
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # 如果达不到数量的话就padding
    # 计算需要填充的数量，以确保检测结果张量的大小为 DETECTION_MAX_INSTANCES  Tensor("mrcnn_detection/sub_6:0", shape=(), dtype=int32)
    '''
    计算了一个差距值 `gap`，它是配置中最大检测实例数 `config.DETECTION_MAX_INSTANCES` 与检测结果张量 `detections` 的形状的第一维（通常是实例的数量）之间的差值。
        1. `config.DETECTION_MAX_INSTANCES`：这是一个配置参数，表示最大检测实例数。
        2. `tf.shape(detections)[0]`：使用 TensorFlow 的 `tf.shape()` 函数获取检测结果张量 `detections` 的形状，并取第一维的值，即实例的数量。
        3. `gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]`：计算最大检测实例数与实际检测实例数之间的差距。
    这个差距值 `gap` 通常用于在需要将检测结果张量填充到固定大小的情况下。例如，如果希望检测结果张量始终具有相同的大小，即使实际检测到的实例数量少于最大数量，那么可以使用 `gap` 值来确定需要填充的额外实例的数量。
    '''
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    # 对检测结果张量进行填充，使其大小为 DETECTION_MAX_INSTANCES  Tensor("mrcnn_detection/Pad:0", shape=(?, 6), dtype=float32)
    '''
    使用 TensorFlow 的 `tf.pad()` 函数对张量 `detections` 进行填充操作:
        1. `detections`：这是要进行填充的张量。
        2. `[(0, gap), (0, 0)]`：这是一个列表，指定了在张量的每个维度上要添加的填充数量。在这个例子中，第一个维度上添加 `gap` 个填充元素，第二个维度上不添加填充元素。
        3. `"CONSTANT"`：这是填充的模式。`CONSTANT` 模式表示使用一个固定的值进行填充。      
    通过执行这段代码，张量 `detections` 的大小将在第一个维度上增加 `gap` 个元素，这些元素将使用固定值进行填充。这样可以确保张量具有指定的大小，以便进行后续的处理或与其他张量进行操作。
    '''
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


'''
定义 `norm_boxes_graph` 的函数，用于将边界框进行归一化处理:
    1. `boxes`：表示要归一化的边界框张量，形状为 `[..., 4]`，其中 `...` 表示任意数量的前导维度。
    2. `shape`：表示图像的形状张量，形状为 `[2]`，其中包含图像的高度 `h` 和宽度 `w`。
    函数的主要步骤如下：
    1. 将 `shape` 张量转换为 `float32` 类型，并使用 `tf.split` 函数将其拆分为高度 `h` 和宽度 `w`。
    2. 计算归一化的比例因子 `scale`，通过将高度和宽度连接起来并减去常量 `1.0`。
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
        rois = inputs[0]           # Tensor("ROI/packed_2:0", shape=(1, ?, 4), dtype=float32)
        mrcnn_class = inputs[1]    # Tensor("mrcnn_class/Reshape_1:0", shape=(?, 1000, 81), dtype=float32)
        mrcnn_bbox = inputs[2]     # Tensor("mrcnn_bbox/Reshape:0", shape=(?, ?, 81, 4), dtype=float32)
        image_meta = inputs[3]     # Tensor("input_image_meta:0", shape=(?, 93), dtype=float32)  1+3+3+4+1+self.NUM_CLASSES(81)

        # 找到window的小数形式: 使用 parse_image_meta_graph 函数解析 image_meta，获取图像的形状信息，并将其赋值给变量 image_shape。
        #                     然后，使用 norm_boxes_graph 函数将 m['window'] 转换为小数形式的窗口，并将其赋值给变量 window。
        '''
        { 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES(81)： 
            1-图像的唯一标识符： 'image_id': <tf.Tensor 'mrcnn_detection/strided_slice:0' shape=(?,) dtype=float32>, 
            3-原始图像的形状： 'original_image_shape': <tf.Tensor 'mrcnn_detection/strided_slice_1:0' shape=(?, 3) dtype=float32>, 
            3-当前图像的形状：  'image_shape': <tf.Tensor 'mrcnn_detection/strided_slice_2:0' shape=(?, 3) dtype=float32>, 
            4-图像在窗口中的位置（以像素为单位）： 'window': <tf.Tensor 'mrcnn_detection/strided_slice_3:0' shape=(?, 4) dtype=float32>, 
            1-图像的缩放比例： 'scale': <tf.Tensor 'mrcnn_detection/strided_slice_4:0' shape=(?,) dtype=float32>, 
            81-活动类别的标识符：   'active_class_ids': <tf.Tensor 'mrcnn_detection/strided_slice_5:0' shape=(?, 81) dtype=float32>}
        '''
        m = parse_image_meta_graph(image_meta)

        # Tensor("mrcnn_detection/strided_slice_6:0", shape=(3,), dtype=float32)
        image_shape = m['image_shape'][0]
        # Tensor("mrcnn_detection/truediv:0", shape=(?, 4), dtype=float32)
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
