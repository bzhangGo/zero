# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_lr(init_lr, global_step, p):

  strategy = p.lrate_strategy.lower()
  
  lr = init_lr
  if strategy == "noam":
    step = tf.to_float(global_step)
    warmup_steps = tf.to_float(p.warmup_steps)
    multiplier = p.hidden_size ** -0.5
    decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                    (step + 1) ** -0.5)

    lr = init_lr * decay
  elif strategy == "none":
    lr = init_lr
  else:
    raise NotImplementedError("{} is not supported".format(strategy))

  lr = tf.maximum(tf.minimum(lr, p.max_lrate), p.min_lrate)

  return lr
