import os
# supress unnecessary logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import tensorflow as tf
from configs import cfg
from engine.trainer import do_train
from engine.builder import make_optimizer
from engine.inference import do_evaluation
from datasets.builder import make_data_loader
from models.detector import build_detection_model
from utils.utils import Logger, setup_dirs, seed_everything


def load_model(model, ckpt):
    rand_inp = tf.random.normal([4, 300, 300, 3])
    # model has to be called before loading the weights
    dummy_out = model(rand_inp)
    model.load_weights(ckpt)
    return model

def train(cfg, logger):
    model = build_detection_model(cfg)
    
    if cfg.CKPT:
        model = load_model(model, cfg.CKPT)
        logger.write(f'Loaded ckpt from {cfg.CKPT}\n')

    train_loader, dataset_len = make_data_loader(cfg, is_train=True)
    optimizer = make_optimizer(cfg, train_loader, dataset_len)
    model = do_train(cfg, model, train_loader, dataset_len, optimizer, logger)
    return model


def main():
    parser = argparse.ArgumentParser(description='SSD Training with tf2.0')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="Whether to test or train the model",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()    

    os.environ['CUDA_VISIBLE_DEVICES']= str(cfg.GPU)

    if cfg.OUTPUT_DIR:
        setup_dirs(cfg)

    logger = Logger()
    logger.open(cfg.OUTPUT_DIR + '/train_log.txt', mode='a')
    logger.write('*'*30)
    logger.write('\n')
    logger.write(f'Logging arguments from config file: {args.config_file}!!\n')

    # freeze everything
    seed_everything()

    # train the model
    if not args.test:
        model = train(cfg, logger)

    # evaluate
    if args.test:
        logger.write(f'Starting Evaluating...!!\n')
        model = build_detection_model(cfg)
    
        if cfg.CKPT:
            model = load_model(model, cfg.CKPT)
            logger.write(f'Loaded ckpt from {cfg.CKPT}\n')

        do_evaluation(cfg, model, logger)
    
    
if __name__ == '__main__':
    main()