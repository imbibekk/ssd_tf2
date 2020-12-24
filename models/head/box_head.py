import tensorflow as tf
from utils import box_utils
from models import registry
from .loss import MultiBoxLoss
from .inference import PostProcessor
from models.head.box_predictor import make_box_predictor
from models.anchors.prior_box import PriorBox


@registry.BOX_HEADS.register('SSDBoxHead')
class SSDBoxHead(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super(SSDBoxHead, self).__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None

    def call(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)  # (batch_size, num_priors, num_C) | (batch_size, num_priors, 4)
        if targets is not None:
            return self._call_train(cls_logits, bbox_pred, targets)
        return self._call_test(cls_logits, bbox_pred)
        
    def _call_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets 
        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
        return reg_loss, cls_loss
        
    def _call_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)()
        scores = tf.keras.activations.softmax(cls_logits, axis=2)
        boxes = box_utils.convert_locations_to_boxes(bbox_pred, self.priors , self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE)
        boxes = box_utils.center_form_to_corner_form(boxes)
        
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections


