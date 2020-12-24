import os
from math import ceil
from tqdm import tqdm
from evaluation import evaluate
from datasets.builder import get_dataset, make_data_loader
import tensorflow as tf

def compute_on_dataset(model, data_loader, steps_per_epoch):
    results_dict = {}
    for images, boxes, labels, image_ids in tqdm(data_loader, total=steps_per_epoch):
        outputs = model(images)
        outputs = [o for o in outputs]
        results_dict.update(
            {int(img_id): result for img_id, result in zip(image_ids, outputs)}
        )
    return results_dict


def do_evaluation(cfg, model, logger):
    dataset_test = get_dataset(cfg, is_train=False)
    data_loader, dataset_len = make_data_loader(cfg, is_train=False)
    steps_per_epoch = ceil(dataset_len / cfg.SOLVER.BATCH_SIZE)
    output_folder = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TEST[0])
    predictions = compute_on_dataset(model, data_loader, steps_per_epoch)
    return evaluate(dataset=dataset_test, predictions=predictions, output_dir=output_folder, logger=logger)
    