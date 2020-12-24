import tensorflow as tf
from utils.nms import batched_nms
from utils.container import Container
import torch


class PostProcessor:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.INPUT.IMAGE_SIZE
        self.height = cfg.INPUT.IMAGE_SIZE

    def __call__(self, detections):
        batches_scores, batches_boxes = detections
        batch_size = batches_scores.shape[0]
        results = []
        for batch_id in range(batch_size):
            scores, boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            num_boxes = scores.shape[0]
            num_classes = scores.shape[1]

            boxes = tf.reshape(boxes, (num_boxes, 1, 4))
            boxes = tf.broadcast_to(boxes, [num_boxes, num_classes, 4])
        
            labels = tf.range(num_classes)
            labels = tf.reshape(labels, (1, num_classes))
            labels = tf.broadcast_to(labels, [num_boxes, num_classes])
            
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = tf.reshape(boxes,(-1, 4))
            scores = tf.reshape(scores, -1)
            labels = tf.reshape(labels, -1)

            # remove low scoring boxes
            indices = (scores > self.cfg.TEST.CONFIDENCE_THRESHOLD)
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

            boxes = boxes.numpy() # convert to numpy bcoz tf doesn't support tensor assignment
            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height
            boxes =  tf.convert_to_tensor(boxes) # convert numpy back to tf-tensor

            keep = batched_nms(boxes, scores, labels, self.cfg.TEST.MAX_PER_IMAGE, self.cfg.TEST.NMS_THRESHOLD)
            
            boxes = tf.gather(boxes, keep)
            scores = tf.gather(scores, keep)
            labels = tf.gather(labels, keep)
            
            container = Container(boxes=boxes, labels=labels, scores=scores)
            container.img_width = self.width
            container.img_height = self.height
            results.append(container)
        return results