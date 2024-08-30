"""
Mask R-CNN
Common utility functions and classes.
"""

import sys
import os
import logging
import math
import random
import skimage
import skimage.transform
import numpy as np
import tensorflow as tf
import scipy
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


# ----------------------------------------------------------#
#  Bounding Boxes
# ----------------------------------------------------------#

def extract_bboxes(mask):
    # 利用语义分割的mask找到包围它的框
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """
        编码运算
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """
        编码运算
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    # 保持原有的image 保存输入图像的数据类型
    image_dtype = image.dtype  # uint8
    # 初始化参数
    h, w = image.shape[:2]  # 获取图像的高度和宽度  h:1330 w:1330
    window = (0, 0, h, w)  # 窗口信息  {tuple:4} (0, 0, 1330, 1330)
    scale = 1  # 缩放比例
    padding = [(0, 0), (0, 0), (0, 0)]  # 填充量
    crop = None  # 裁剪区域

    # 如果模式为 none，则直接返回图像、窗口、缩放比例、填充量和裁剪区域
    if mode == "none":
        return image, window, scale, padding, crop

    # 计算变化的尺度
    if min_dim:  # 如果设置了最小边长
        scale = max(1, min_dim / min(h, w))  # 计算缩放比例，确保图像的最小边长不小于 min_dim    1
    if min_scale and scale < min_scale:  # 如果设置了最小缩放比例且当前缩放比例小于最小缩放比例
        scale = min_scale  # 将缩放比例设置为最小缩放比例

    # 判断按照原来的尺寸缩放是否会超过最大边长
    if max_dim and mode == "square":  # 如果设置了最大边长且模式为 square
        image_max = max(h, w)  # 计算图像的最大边长  1330
        if round(image_max * scale) > max_dim:  # 如果按照当前缩放比例缩放后的最大边长超过了最大边长限制  1330 > 1024
            scale = max_dim / image_max  # 计算新的缩放比例，确保缩放后的最大边长不超过最大边长限制   0.7699248120300752

    # 对图片进行resize
    if scale != 1:  # 如果缩放比例不为 1
        # 使用 resize 函数对图像进行缩放，preserve_range=True 表示保持像素值的范围  round() 函数用于对浮点数进行四舍五入操作，并返回最接近的整数
        image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)  # ndarray:(1024,1024,3)

    # 是否需要padding填充
    if mode == "square":
        # 计算四周padding的情况  获取缩放后的图像高度和宽度
        h, w = image.shape[:2]  # h:1024  w:1024

        top_pad = (max_dim - h) // 2  # 计算顶部填充量  0
        bottom_pad = max_dim - h - top_pad  # 计算底部填充量   0
        left_pad = (max_dim - w) // 2  # 计算左侧填充量   0
        right_pad = max_dim - w - left_pad  # 计算右侧填充量   0

        # 向四周进行填充 设置填充量  [(0, 0), (0, 0), (0, 0)]
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        # 使用 np.pad 函数对图像进行填充  (1024,1024,3)
        image = np.pad(image, padding, mode='constant', constant_values=0)
        # 更新窗口信息  (0, 0, 1024, 1024)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)

    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    # 将mask按照scale放大缩小后
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """
    减少语义分割载入时的size
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass

# 将神经网络生成的掩码转换为与原始图像形状相似的格式
def unmold_mask(mask, bbox, image_shape):
    """
    Converts a mask generated by the neural network to a format similar to its original shape.

    mask: [height, width] of type float. A small, typically 28x28 mask.  输入的掩码，形状为 [height, width]，类型为 float。这是一个较小的掩码，通常为 28x28
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.  输入的边界框，形状为 [y1, x1, y2, x2]

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    # 提取边界框的坐标
    y1, x1, y2, x2 = bbox
    # 将掩码调整大小，使其与边界框的尺寸匹配
    mask = resize(mask, (y2 - y1, x2 - x1))
    # 将掩码中的值与阈值进行比较，大于等于阈值的设置为 1，小于阈值的设置为 0，并将结果转换为布尔类型
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    # 创建一个与原始图像形状相同的全零掩码
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    # 调整大小后的掩码放置在全零掩码的相应位置上
    full_mask[y1:y2, x1:x2] = mask
    # 函数返回转换后的二进制掩码，与原始图像具有相同的尺寸
    return full_mask


# ----------------------------------------------------------#
#  Miscellaneous
# ----------------------------------------------------------#

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = \
            compute_ap(gt_box, gt_class_id, gt_mask,
                       pred_box, pred_class_id, pred_score, pred_mask,
                       iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# batch_slice 函数的作用是将批量数据分成多个切片，并对每个切片应用相同的计算图(graph_fn: lambda x, y: tf.gather(x, y),为调用时传入的不同函数)，
#             最后将结果合并成列表 ([(scores1, ix1,(scores2, ix2),...)]), 这样可以在一次计算中处理多个数据样本，提高计算效率
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """
    Splits inputs into slices and feeds each slice to a copy of the given computation graph and then combines the results.
    It allows you to run a graph on a batch of inputs even if the graph is written to support one instance only.

    inputs: list of tensors. All must have the same first dimension length
         inputs（输入张量列表）  [scores, ix]  -> Tensor("ROI/strided_slice:0", shape=(?, ?), dtype=float32)、  Tensor("ROI/top_anchors:1", shape=(?, ?), dtype=int32)
    graph_fn: A function that returns a TF tensor that's part of a graph.     计算图函数
    batch_size: number of slices to divide the data into.                     切片数量 1
    names: If provided, assigns names to the resulting tensors.               结果张量的名称列表
    """
    if not isinstance(inputs, list):  # 函数检查输入是否为列表，如果不是，则将其转换为列表
        inputs = [inputs]
        # [scores, ix]  -> Tensor("ROI/strided_slice:0", shape=(?, ?), dtype=float32)、
        #                  Tensor("ROI/top_anchors:1", shape=(?, ?), dtype=int32)
    outputs = []  # 创建一个空列表 outputs 用于存储每个切片的结果
    for i in range(batch_size):  # 循环
        '''
        将输入数据按照切片数量进行分割，得到每个切片的输入数据 inputs_slice  
        [<tf.Tensor 'ROI/strided_slice_2:0' shape=(?,) dtype=float32>, <tf.Tensor 'ROI/strided_slice_3:0' shape=(?,) dtype=int32>]'''
        inputs_slice = [x[i] for x in inputs]
        '''调用 graph_fn 函数对每个切片的输入数据进行计算，得到输出结果 output_slice  Tensor("ROI/GatherV2:0", shape=(?,), dtype=float32)'''
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices   outputs [[<tf.Tensor 'ROI/GatherV2:0' shape=(?,) dtype=float32>]]
    '''
    `*outputs` 是 Python 中的一种特殊语法，用于将列表或元组中的元素作为独立的参数传递给函数。
        这里，`outputs` 是一个列表，通过使用 `*outputs`，你可以将列表中的每个元素作为独立的参数传递给 `zip()` 函数。
     `zip()` 是 Python 中的一个内置函数，用于将多个可迭代对象（如列表、元组等）组合成一个元组序列：
            zip(iterable1, iterable2,...)  其中，`iterable1, iterable2,...` 是要组合的可迭代对象。`zip()` 函数会返回一个迭代器，该迭代器可以逐个访问组合后的元组。     
                下面是一个简单的示例，展示了如何使用 `zip()` 函数：
                    numbers = [1, 2, 3]
                    letters = ['a', 'b', 'c']
                    zipped = zip(numbers, letters)      # 组合成了一个元组序列。每个元组包含了一个数字和一个对应的字母(如：1对应a...)
                    # 可以通过迭代访问组合后的元组
                        for item in zipped:
                            print(item)                 # (1, 'a')   (2, 'b')   (3, 'c')
                上面的示例中，`zip(numbers, letters)` 将数字列表和字母列表组合成一个元组序列。通过迭代访问这个元组序列，可以得到每个数字和对应的字母组成的元组。
            需要注意的是，`zip()` 函数返回的是一个迭代器，而不是一个具体的列表或元组。如果需要将组合后的元组序列转换为列表或元组，可以使用 `list()` 或 `tuple()` 函数进行转换。
            此外，`zip()` 函数的长度是由最短的可迭代对象决定的。如果可迭代对象的长度不一致，那么 `zip()` 函数会在最短的可迭代对象结束时停止。     
        这里,它会将 `outputs` 列表中的每个元素作为一个独立的可迭代对象，并将它们组合成一个元组序列。
        这样做的效果是将 `outputs` 列表中的元素按照顺序进行配对，形成一系列的元组。每个元组包含了原来列表中对应位置的元素。
    list() 是 Python 中的一个内置函数，用于创建一个新的列表对象。它可以接受一个可迭代对象作为参数，并将其元素添加到新创建的列表中。
    '''
    outputs = list(zip(*outputs))  # 对 outputs 列表进行转置，将每个输出结果的切片组合在一起

    if names is None:  # 如果提供了名称列表 names，则为每个结果张量设置名称
        names = [None] * len(outputs)  # [None] = [None] * 1

    # 使用 tf.stack 函数将每个输出结果的切片堆叠在一起，得到最终的结果张量  [<tf.Tensor 'ROI/packed:0' shape=(1, ?) dtype=float32>]
    '''
    tf.stack() 是 TensorFlow 中的一个函数，用于沿着指定的轴堆叠张量列表。它接受一个张量列表作为输入，并返回一个通过沿着指定轴堆叠输入张量而形成的单个张量
       使用 `tf.stack()` 函数堆叠张量时，要求待堆叠的张量在除了指定的堆叠轴之外的其他维度上具有相同的形状。
       例如，如果要沿着第一个轴堆叠两个张量 `tensor1` 和 `tensor2`，那么它们在第二个和第三个维度上的形状必须相同。
            如果张量的维度不一致，可能会导致以下错误：ValueError: Dimensions must be equal, but are 3 and 2 for 'stack' (op: 'Stack') with input shapes: [2,3], [2,2].
       为了确保张量可以正确堆叠，你可以在堆叠之前对它们进行预处理，例如通过填充或裁剪使其具有相同的形状。
    举例：
           tensor1 = [[1, 2, 3], [4, 5, 6]]  tensor2  = [[7, 8, 9],   [10, 11, 12]]
           tensors = [tensor1 ，tensor2]
           stacked_tensor = tf.stack(tensors, axis=0)
        在这个例子中，`tensor1` 和 `tensor2` 是两个 2x3 的张量。然后，我们将这两个张量放在一个列表 `tensors` 中。
        接下来，使用 `tf.stack()` 函数将 `tensors` 列表中的张量沿着第一个轴（`axis=0`）进行堆叠。
        这将创建一个新的张量 `stacked_tensor`，它的形状为 2x2x3，其中第一个维度是原来的两个张量，第二个维度是原来的行数，第三个维度是原来的列数。        
        ```
            [[[1, 2, 3],
              [4, 5, 6]],
            
             [[7, 8, 9],
              [10, 11, 12]]]
        ```
        这就是 `tf.stack()` 函数的基本用法。通过指定不同的轴，你可以在不同的维度上进行堆叠操作。
    '''
    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]
    if len(result) == 1:  # 如果结果张量只有一个，则直接返回该张量，否则返回列表形式的结果张量  Tensor("ROI/packed:0", shape=(1, ?), dtype=float32)
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


# 将边界框的坐标从像素坐标转换为归一化坐标
def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.    将边界框的坐标从像素坐标转换为归一化坐标
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates    输入的边界框坐标，形状为 [N, (y1, x1, y2, x2)]，其中 N 是边界框的数量，(y1, x1, y2, x2) 是每个边界框的左上角和右下角的坐标
    shape: [..., (height, width)] in pixels   其中 ... 表示任意数量的维度，(height, width) 是图像的高度和宽度。

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    # 获取图像的高度和宽度
    h, w = shape
    # 创建一个缩放因子数组，用于将像素坐标转换为归一化坐标
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    # 创建一个偏移量数组，用于将边界框的坐标从像素坐标转换为归一化坐标
    shift = np.array([0, 0, 1, 1])
    # 将边界框的坐标减去偏移量，然后除以缩放因子，得到归一化坐标。最后将结果转换为 np.float32 类型并返回
    # 在归一化坐标中，坐标的取值范围是 [0, 1]，其中 (0, 0) 表示图像的左上角，(1, 1) 表示图像的右下角
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)

# 定义了一个名为 resize 的函数，它是对 Scikit-Image 库中的 resize 函数的封装
def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.

            image：要调整大小的图像。
            output_shape：目标输出形状。
            order：插值的阶数，默认为 1。
            mode：填充模式，默认为 'constant'，表示用常数填充。
            cval：用于 'constant' 模式的填充值，默认为 0。
            clip：是否裁剪输出图像，默认为 True。
            preserve_range：是否保留输入图像的范围，默认为 False。
            anti_aliasing：是否进行抗锯齿处理，默认为 False。
            anti_aliasing_sigma：抗锯齿处理的 sigma 值，默认为 None。
    """
    # 首先检查 Scikit-Image 的版本，如果版本大于或等于 0.14
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        # 则使用 Scikit-Image 0.14 及更高版本的参数调用 resize 函数
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:  # 否则，使用旧版本的参数调用 resize 函数
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def mold_image(images, config):
    """
    Expects an RGB image (or array of images) and subtracts the mean pixel and converts it to float. Expects image colors in RGB order.
    实现了对输入图像的预处理，将图像转换为 np.float32 类型，并减去均值像素值：
          images.astype(np.float32)：将输入的图像转换为 np.float32 类型
          config.MEAN_PIXEL：获取配置中的均值像素值  [123.7, 116.8, 103.9]
          images.astype(np.float32) - config.MEAN_PIXEL：将图像的每个像素值减去均值像素值
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

# 创建一个包含上述所有信息(连接到一起) 的 numpy 数组   ndarray:(93,)
def compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids):
    """
    Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.  -> 图像的整数 ID，有助于调试
    original_image_shape: [H, W, C] before resizing or padding.  -> 原始图像的形状，格式为[H, W, C]，其中 H 为高度，W 为宽度，C 为通道数
    image_shape: [H, W, C] after resizing and padding  -> 处理后的图像形状，格式与original_image_shape相同
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)           -> 图像的窗口，以像素为单位，格式为(y1, x1, y2, x2)，表示图像中实际图像的区域（不包括填充）
    scale: The scaling factor applied to the original image (float32)  -> 应用于原始图像的缩放因子（float32 类型）
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets. -> 数据集提供的有效类 ID 列表。如果在多个数据集上进行训练，并且并非所有类都存在于所有数据集中，那么这个参数会很有用
    """
    # 创建一个包含上述所有信息的 numpy 数组     + 运算符在这里的作用是连接数组，而不是简单的加法运算。它将各个部分按照顺序连接在一起，形成一个新的数组
    meta = np.array(
        [image_id] +  # size=1  图像ID->0
        list(original_image_shape) +  # size=3  原始图像的形状->(1330,1330,3)
        list(image_shape) +  # size=3  处理后的图像形状->(1024,1024,3)
        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates   图像的窗口->(0, 0, 1024, 1024)
        [scale] +  # size=1  缩放因子->0.7699248120300752
        list(active_class_ids)  # size=num_classes  分类ID列表->ndarray(81,)
    )
    return meta  # 返回组合好的图像元数据数组

# 图像预处理
def mold_inputs(config, images):
    # 创建一个空列表 molded_images，用于存储处理后的图像
    molded_images = []
    # 创建一个空列表 image_metas，用于存储图像的元数据
    image_metas = []
    # 创建一个空列表 windows，用于存储图像的窗口信息
    windows = []
    # 遍历输入的图像列表
    for image in images:
        # Resize image  使用 resize_image 函数对图像进行缩放，返回处理后的图像、窗口、缩放比例、填充量和裁剪区域
        # molded_image->ndarray(1024,1024,3)  window->(0, 0, 1024, 1024)  scale->0.7699248120300752  padding->[(0, 0), (0, 0), (0, 0)] crop->None
        molded_image, window, scale, padding, crop = resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,  # 1024
            min_scale=config.IMAGE_MIN_SCALE,  # 0
            max_dim=config.IMAGE_MAX_DIM,  # 1024
            mode=config.IMAGE_RESIZE_MODE)  # square
        # 使用 mold_image 函数对resize后的图像进行进一步处理, 将图像转换为 np.float32 类型，并减去均值像素值
        molded_image = mold_image(molded_image, config)
        # Build image_meta(入参所有信息连接到一起) 使用 compose_image_meta 函数构建图像的元数据  ndarray:(93,)  [0.00000000e+00 1.33000000e+03 1.33000000e+03 3.00000000e+00, ...
        image_meta = compose_image_meta(
            0, image.shape, molded_image.shape, window, scale,
            np.zeros([config.NUM_CLASSES], dtype=np.int32))
        # Append
        # 将处理后的图像添加到 molded_images 列表中
        molded_images.append(molded_image)
        # 将图像的窗口信息添加到 windows 列表
        windows.append(window)
        # 将图像的元数据添加到 image_metas 列表中
        image_metas.append(image_meta)
    # Pack into arrays
    # 将 molded_images 列表转换为 NumPy 数组    (1,1024,1024,3)
    molded_images = np.stack(molded_images)
    # 将 image_metas 列表转换为 NumPy 数组   (1,93)
    image_metas = np.stack(image_metas)
    # 将 windows 列表转换为 NumPy 数组  (0,0,1024,1024)
    windows = np.stack(windows)
    # 返回处理后的图像数组、图像元数据数组和窗口信息数组
    return molded_images, image_metas, windows


"""
  对目标检测模型的输出结果进行处理:
        包括提取边界框、类别 ID、得分和掩码，并进行一些坐标转换和数据处理操作。
        最终返回处理后的结果，以便后续的分析或应用
"""
def unmold_detections(detections, mrcnn_mask, original_image_shape, image_shape, window):
    # 找到检测结果中类别 ID 为 0 的索引
    zero_ix = np.where(detections[:, 4] == 0)[0]
    # 根据类别 ID 为 0 的索引数量，确定要处理的检测结果数量
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # 提取前 N 个检测结果的边界框信息
    boxes = detections[:N, :4]
    # 提取前 N 个检测结果的类别 ID，并转换为整数类型
    class_ids = detections[:N, 4].astype(np.int32)
    # 提取前 N 个检测结果的得分
    scores = detections[:N, 5]
    # 根据类别 ID 提取对应的掩码
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]
    # 对窗口进行归一化处理，使其适应图像的尺寸
    window = norm_boxes(window, image_shape[:2])

    # 获取归一化窗口的坐标
    wy1, wx1, wy2, wx2 = window
    # 计算窗口的偏移量
    shift = np.array([wy1, wx1, wy1, wx1])
    # 计算窗口的高度
    wh = wy2 - wy1  # window height
    # 计算窗口的宽度
    ww = wx2 - wx1  # window width

    # 创建缩放因子数组
    scale = np.array([wh, ww, wh, ww])
    # 将边界框的坐标进行归一化
    boxes = np.divide(boxes - shift, scale)
    # 将归一化的边界框坐标转换回原始图像的坐标
    boxes = denorm_boxes(boxes, original_image_shape[:2])

    # 找到边界框面积小于等于 0 的索引
    exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    # 如果边界框面积小于等于 0 的索引数量大于 0，则执行以下操作
    if exclude_ix.shape[0] > 0:
        # 删除面积小于等于 0 的边界框
        boxes = np.delete(boxes, exclude_ix, axis=0)
        # 删除对应的类别 ID
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        # 删除对应的得分
        scores = np.delete(scores, exclude_ix, axis=0)
        # 删除对应的掩码
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    # 创建一个空列表，用于存储展开的掩码(掩码与原始图像形状相似)
    full_masks = []
    # 遍历处理后的检测结果
    for i in range(N):
        # 对每个掩码进行展开操作： 将神经网络生成的掩码转换为与原始图像形状相似的格式
        full_mask = unmold_mask(masks[i], boxes[i], original_image_shape)
        # 将展开后的掩码添加到列表中
        full_masks.append(full_mask)

    # 如果有展开的掩码，则将它们堆叠在一起，否则创建一个空的掩码数组
    '''
    在对 `full_masks` 进行处理。它的作用是根据 `full_masks` 的情况进行不同的操作:
        - `np.stack(full_masks, axis=-1)`：这是将 `full_masks` 中的多个掩码沿着最后一个维度（在这里是 `axis=-1`）进行堆叠。如果 `full_masks` 不为空，那么执行这个操作。
        - `np.empty(original_image_shape[:2] + (0,))`：这是创建一个与 `original_image_shape[:2]` 相同形状的空数组，但是最后一个维度的大小为 `0`。如果 `full_masks` 为空，那么执行这个操作。   
    综上所述，这段代码的目的是根据 `full_masks` 的情况，创建一个合适的数组，该数组的形状与 `original_image_shape[:2]` 相同，但是最后一个维度的大小可能为 `0` 或者是 `full_masks` 中掩码的数量。
    '''
    full_masks = np.stack(full_masks, axis=-1) \
        if full_masks else np.empty(original_image_shape[:2] + (0,))

    # 函数返回处理后的边界框、类别 ID、得分和展开的掩码
    return boxes, class_ids, scores, full_masks


def norm_boxes_graph(boxes, shape):
    """
        用于进行标准化，限制到0-1之间
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def parse_image_meta_graph(meta):
    """
        对输入的meta进行拆解
        将包含图像属性的张量解析为其组件。
        返回解析的张量的dict。
    """
    image_id = meta[:, 0]  # 图片的id
    original_image_shape = meta[:, 1:4]  # 原始的图片的大小
    image_shape = meta[:, 4:7]  # resize后图片的大小
    window = meta[:, 7:11]  # (y1, x1, y2, x2)有效的区域在图片中的位置
    scale = meta[:, 11]  # 长宽的变化状况
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }
