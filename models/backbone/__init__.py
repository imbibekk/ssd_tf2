from models import registry
from .vgg import VGG

__all__ = ['build_backbone', 'VGG']

def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)