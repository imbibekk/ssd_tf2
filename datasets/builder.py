import tensorflow as tf

from .voc import VOCDataset
from .coco import COCODataset
from .path_catlog import DatasetCatalog
from .target_transform import SSDTargetTransform
from .image_transform import *
from models.anchors.prior_box import PriorBox

_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
}


def build_dataset(DATA_DIR, dataset_list, transform=None, target_transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(DATA_DIR, dataset_name)
        args = data['args']
        factory = _DATASETS[data['factory']]
        args['transform'] = transform
        args['target_transform'] = target_transform
        if factory == VOCDataset:
            args['keep_difficult'] = not is_train
        elif factory == COCODataset:
            args['remove_empty'] = is_train
        dataset = factory(**args)
        datasets.append(dataset)
    return datasets
    

def build_image_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
        ]
    else:
        transform = [
            ConvertFromInts(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform


def make_data_loader(cfg, is_train=True):
    train_transform = build_image_transforms(cfg, is_train=is_train)
    target_transform = build_target_transform(cfg)
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(cfg.DATA_DIR, dataset_list, transform=train_transform, target_transform=target_transform, is_train=is_train)
    dataset_len = 0
    for dataset in datasets:
        dataset_len += dataset.__len__()

    for idx, dataset in enumerate(datasets):
        if idx == 0:
            loader= tf.data.Dataset.from_generator(dataset.generate, (tf.float32, tf.float32, tf.int64, tf.int64))
        else:
            loader = loader.concatenate(tf.data.Dataset.from_generator(dataset.generate, (tf.float32, tf.float32, tf.int64, tf.int64)))
    loader = loader.prefetch(cfg.SOLVER.BATCH_SIZE)
    loader = loader.batch(cfg.SOLVER.BATCH_SIZE)
    
    if is_train:
        loader = loader.shuffle(40)
    return loader.take(-1), dataset_len


def get_dataset(cfg, is_train=False):
    train_transform = build_image_transforms(cfg, is_train=is_train)
    target_transform = None 
    dataset = build_dataset(cfg.DATA_DIR, cfg.DATASETS.TEST, transform=train_transform, target_transform=None, is_train=False)[0]
    return dataset
