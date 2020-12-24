
import re
import math
from typing import Callable, List, Optional, Union

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.
    Args:
        initial_learning_rate (:obj:`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule_fn (:obj:`Callable`):
            The schedule function to apply after the warmup for the rest of training.
        warmup_steps (:obj:`int`):
            The number of steps for the warmup part of training.
        power (:obj:`float`, `optional`, defaults to 1):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (:obj:`str`, `optional`):
            Optional name prefix for the returned tensors during the schedule.
    """

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }



def make_optimizer(cfg, data_loader, dataset_len):
    steps_per_epoch = math.ceil(dataset_len / cfg.SOLVER.BATCH_SIZE)
    num_train_steps = steps_per_epoch * cfg.TOTAL_EPOCHS 
    num_warmup_steps = cfg.SOLVER.WARMUP_ITERS 
    
    lr_schedule = PiecewiseConstantDecay(
        boundaries=[int(steps_per_epoch * cfg.SOLVER.LR_EPOCHS[0]),  
                    int(steps_per_epoch * cfg.SOLVER.LR_EPOCHS[1])],
        values=[cfg.SOLVER.LR, cfg.SOLVER.LR * 0.1, cfg.SOLVER.LR * 0.01])
    
    lr_schedule = WarmUp(
            initial_learning_rate=cfg.SOLVER.LR,
            decay_schedule_fn=lr_schedule,
            warmup_steps=num_warmup_steps,
        )

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule,
                                        momentum=cfg.SOLVER.MOMENTUM)

    return optimizer

    