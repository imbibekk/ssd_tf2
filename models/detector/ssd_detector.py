from tensorflow.keras import Model
from models.backbone import build_backbone
from models.head import build_box_head

class SSDDetector(Model):
    def __init__(self, cfg):
        super(SSDDetector, self).__init__()
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)

    def call(self, x, targets=None):
        features = self.backbone(x)
        outputs = self.box_head(features, targets)
        return outputs

