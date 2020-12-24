import os
from tqdm import tqdm
from math import ceil
import tensorflow as tf
from engine.inference import do_evaluation

def do_train(cfg, model, data_loader, dataset_len, optimizer, logger):
    
    logger.write('Start training!!!\n')
        
    def _train_one_epoch():
        cls_loss_meter = tf.keras.metrics.Mean('cls_loss', dtype=tf.float32)
        bbox_loss_meter = tf.keras.metrics.Mean('bbox_loss', dtype=tf.float32)
        avg_loss_meter = tf.keras.metrics.Mean('avg_loss', dtype=tf.float32)

        steps_per_epoch = ceil(dataset_len / cfg.SOLVER.BATCH_SIZE)
        for images, boxes, labels, index in tqdm(data_loader, total=steps_per_epoch):
            with tf.GradientTape() as tape:
                targets =  (boxes, labels)
                bbox_loss, class_loss = model(images, targets) # reg_loss, cls_loss
                loss = class_loss + bbox_loss
                l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
                l2_loss = cfg.SOLVER.WEIGHT_DECAY * tf.math.reduce_sum(l2_loss)
                loss += l2_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Record loss and model accuracy.
            cls_loss_meter.update_state(class_loss)
            bbox_loss_meter.update_state(bbox_loss)
            avg_loss_meter.update_state(loss)
        return cls_loss_meter.result().numpy(), bbox_loss_meter.result().numpy(), avg_loss_meter.result().numpy()

    best_mAP = 0.0
    patience = 0.0
    for epoch in range(cfg.TOTAL_EPOCHS):

        cls_loss, bbox_loss, avg_loss = _train_one_epoch()
        logger.write(f'[Train]Epoch: {epoch} | CLS Loss: {cls_loss} | BBOX Loss: {bbox_loss} | AVG Loss: {avg_loss}\n')

        logger.write('Start Evaluating!!!\n')
        mAP = do_evaluation(cfg, model, logger)
        if mAP > best_mAP:
            best_mAP = mAP
            model.save_weights(os.path.join(cfg.OUTPUT_DIR, 'model_weights.h5'))
            
        logger.write(f'Best mAP: {best_mAP}\n')
    