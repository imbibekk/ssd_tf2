import sys
import warnings
import tensorflow as tf


def nms(boxes, scores, top_k, nms_thresh):
    """ Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
        boxes(Tensor[N, 4]): boxes in (x1, y1, x2, y2) format, use absolute coordinates(or relative coordinates)
        scores(Tensor[N]): scores
        nms_thresh(float): thresh
    Returns:
        indices kept.
    """
    keep_idx = tf.image.non_max_suppression(tf.cast(boxes, tf.float32), tf.cast(scores, tf.float32), top_k, nms_thresh) #.numpy()
    return keep_idx


def batched_nms(boxes, scores, idxs, top_k, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold
    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if tf.size(boxes) == 0:
        return tf.zeors([0,])

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = tf.math.reduce_max(boxes)
    offsets = tf.cast(idxs, boxes.dtype) * (max_coordinate + 1.0) 
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, top_k, iou_threshold)
    return keep