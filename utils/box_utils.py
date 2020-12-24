import tensorflow as tf
import math
import numpy as np

def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_mask, tf.int32), axis=1, keepdims=True) #pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    #print('loss', loss.shape, 'pos_mask', pos_mask.shape)
    #loss[pos_mask] = -math.inf
    loss = loss.numpy()
    loss[pos_mask.numpy()] = -math.inf #-1e6
    loss = tf.convert_to_tensor(loss)

    #loss = tf.tensor_scatter_nd_update(
    #    loss,
    #    pos_mask,
    #    tf.range(best_default_idx.shape[0], dtype=tf.int64))

    #loss[pos_mask.numpy()] = -100000000000
    #_, indexes = loss.sort(dim=1, descending=True)
    #_, orders = indexes.sort(dim=1)
    #neg_mask = orders < num_neg
    rank = tf.argsort(loss, axis=1, direction='DESCENDING')
    rank = tf.argsort(rank, axis=1)
    
    neg_mask = rank < num_neg #tf.expand_dims(num_neg, 1)
    #print(pos_mask.shape, neg_mask.shape, rank.shape)
    return tf.math.logical_or(pos_mask, neg_mask)
    #return pos_mask | neg_mask


def corner_form_to_center_form(boxes):
    """ Transform boxes of format (xmin, ymin, xmax, ymax)
        to format (cx, cy, w, h)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    """
    center_box = tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]], axis=-1)

    return center_box


def center_form_to_corner_form(boxes):
    """ Transform boxes of format (cx, cy, w, h)
        to format (xmin, ymin, xmax, ymax)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    corner_box = tf.concat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    return corner_box



def compute_area(top_left, bot_right):
    """ Compute area given top_left and bottom_right coordinates
    Args:
        top_left: tensor (num_boxes, 2)
        bot_right: tensor (num_boxes, 2)
    Returns:
        area: tensor (num_boxes,)
    """
    # top_left: N x 2
    # bot_right: N x 2
    hw = tf.clip_by_value(bot_right - top_left, 0.0, 512.0)
    area = hw[..., 0] * hw[..., 1]

    return area


def compute_iou(boxes_a, boxes_b, eps=1e-5):
    """ Compute overlap between boxes_a and boxes_b
    Args:
        boxes_a: tensor (num_boxes_a, 4)
        boxes_b: tensor (num_boxes_b, 4)
    Returns:
        overlap: tensor (num_boxes_a, num_boxes_b)
    """
    # boxes_a => num_boxes_a, 1, 4
    boxes_a = tf.expand_dims(boxes_a, 1)

    # boxes_b => 1, num_boxes_b, 4
    boxes_b = tf.expand_dims(boxes_b, 0)
    top_left = tf.math.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bot_right = tf.math.minimum(boxes_a[..., 2:], boxes_b[..., 2:])

    overlap_area = compute_area(top_left, bot_right)
    area_a = compute_area(boxes_a[..., :2], boxes_a[..., 2:])
    area_b = compute_area(boxes_b[..., :2], boxes_b[..., 2:])

    overlap = overlap_area / (area_a + area_b - overlap_area + eps)

    return overlap


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)
    

def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    #print('center_form_boxes: ', center_form_boxes.shape, 'center_form_priors', center_form_priors.shape)
    # priors can have one dimension less
    return tf.concat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / (center_form_priors[..., 2:] * center_variance),
        tf.math.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], axis=- 1)



def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    return tf.concat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2], 
        tf.math.exp(locations[...,2:] * size_variance) * priors[...,2:]
        ], axis=-1)
    


'''
def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    """Assign ground truth boxes and targets to priors.
    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels
'''



def assign_priors(gt_boxes, gt_labels, corner_form_priors, iou_threshold=0.5):
    """Assign ground truth boxes and targets to priors.
    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # Convert default boxes to format (xmin, ymin, xmax, ymax)
    # in order to compute overlap with gt boxes
    #transformed_default_boxes = transform_center_to_corner(default_boxes)
    
    #print(gt_boxes.shape, gt_labels.shape, corner_form_priors.shape)  # (3, 4) (3,) (8732, 4)
    iou = compute_iou(corner_form_priors, gt_boxes)
    '''
    # size: num_priors
    best_target_per_prior = tf.math.reduce_max(iou, 1)
    best_target_per_prior_index = tf.math.argmax(iou, 1)
    # size: num_targets
    best_prior_per_target = tf.math.reduce_max(iou, 0)
    best_prior_per_target_index = tf.math.argmax(iou, 0)

    best_prior_per_target_index = tf.tensor_scatter_nd_update(
        best_prior_per_target_index,
        tf.expand_dims(best_prior_per_target_index, 1),
        tf.range(best_prior_per_target_index.shape[0], dtype=tf.int64))

    best_target_per_prior = tf.tensor_scatter_nd_update(
        best_target_per_prior,
        tf.expand_dims(best_prior_per_target_index, 1),
        tf.ones_like(best_prior_per_target_index, dtype=tf.float32))

    labels = tf.gather(gt_labels, best_target_per_prior_index)
    
    boxes = tf.gather(gt_boxes, best_target_per_prior_index)
    labels = tf.where(
        tf.less(best_target_per_prior, iou_threshold),
        tf.zeros_like(labels),
        labels)
    '''
    best_gt_iou = tf.math.reduce_max(iou, 1)
    best_gt_idx = tf.math.argmax(iou, 1)

    best_default_iou = tf.math.reduce_max(iou, 0)
    best_default_idx = tf.math.argmax(iou, 0)

    best_gt_idx = tf.tensor_scatter_nd_update(
        best_gt_idx,
        tf.expand_dims(best_default_idx, 1),
        tf.range(best_default_idx.shape[0], dtype=tf.int64))

    best_gt_iou = tf.tensor_scatter_nd_update(
        best_gt_iou,
        tf.expand_dims(best_default_idx, 1),
        tf.ones_like(best_default_idx, dtype=tf.float32))

    #print(gt_labels.shape, best_gt_idx.shape)
    gt_confs = tf.gather(gt_labels, best_gt_idx)
    gt_confs = tf.where(
        tf.less(best_gt_iou, iou_threshold),
        tf.zeros_like(gt_confs),
        gt_confs)

    gt_boxes = tf.gather(gt_boxes, best_gt_idx)
    return gt_boxes, gt_confs



