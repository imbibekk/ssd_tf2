import os
import numpy as np

from .eval_detection_voc import eval_detection_voc


def voc_evaluation(dataset, predictions, output_dir, logger):
    class_names = dataset.class_names

    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxes_list = []
    gt_labels_list = []
    gt_difficults = []
    
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, gt_labels, is_difficult = annotation
        gt_labels_list.append(gt_labels)
        gt_difficults.append(is_difficult.astype(np.bool))

        prediction = predictions[i]
        input_width, input_height = prediction.img_width, prediction.img_height
        img_info = dataset.get_img_info(i)
        real_width, real_height = img_info['width'], img_info['height']

        # rescale ground truth boxes
        gt_boxes[:, 0::2] *= (input_width / real_width)
        gt_boxes[:, 1::2] *= (input_height / real_height)
        gt_boxes_list.append(gt_boxes)

        boxes, labels, scores = prediction['boxes'].numpy(), prediction['labels'].numpy(), prediction['scores'].numpy()
        pred_boxes_list.append(boxes)
        pred_labels_list.append(labels)
        pred_scores_list.append(scores)
    
    result = eval_detection_voc(pred_bboxes=pred_boxes_list,
                                pred_labels=pred_labels_list,
                                pred_scores=pred_scores_list,
                                gt_bboxes=gt_boxes_list,
                                gt_labels=gt_labels_list,
                                gt_difficults=gt_difficults,
                                iou_thresh=0.5,
                                use_07_metric=True)
    
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        logger.write(f'AP_{class_names[i]} : {ap}\n')
    mAP = result['map']
    logger.write(f'mAP : {mAP}\n')
    return mAP