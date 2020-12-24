import tensorflow as tf
from models import registry
from tensorflow.keras import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import L2

class BoxPredictor(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super(BoxPredictor, self).__init__()
        self.cfg = cfg
        self.cls_headers = []
        self.reg_headers = []
        for level, (boxes_per_location, out_channels) in enumerate(zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUT_CHANNELS)):
            self.cls_headers.append(self.cls_block(level, out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
        
    def cls_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reg_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def call(self, features):
        cls_logits = []
        bbox_pred = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            cls_logits.append(cls_header(feature))
            bbox_pred.append(reg_header(feature))
        
        batch_size = features[0].shape[0]
        cls_logits = tf.concat([tf.reshape(c, (c.shape[0], -1)) for c in cls_logits], axis=1)
        bbox_pred = tf.concat([tf.reshape(l, (l.shape[0], -1)) for l in bbox_pred], axis=1)
        cls_logits = tf.reshape(cls_logits, (batch_size, -1, self.cfg.MODEL.NUM_CLASSES))
        bbox_pred = tf.reshape(bbox_pred, (batch_size, -1, 4))
        return cls_logits, bbox_pred


@registry.BOX_PREDICTORS.register('SSDBoxPredictor')
class SSDBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        return layers.Conv2D(boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, padding='same')
        
    def reg_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        return layers.Conv2D(boxes_per_location * 4, kernel_size=3, padding='same')
     
def make_box_predictor(cfg):
    return registry.BOX_PREDICTORS[cfg.MODEL.BOX_HEAD.PREDICTOR](cfg)