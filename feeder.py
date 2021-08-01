# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config
import tensorflow as tf

# define placeholders for feature feeding
# notice that this part is also closely related with data loading
# the naming part should be matched with each other

# add redundant placeholders here, be a super set of the data part
# anyway, feeding redundant to devices is safe


class Feeder(object):
  def __init__(self, num_devices, is_train=True, is_score=False):
    self._num_devices = max(1, num_devices)
    self._placeholders = None
    self._is_train = is_train
    self._is_score = is_score

  def get_placeholders(self):

    def _get():
      placeholders = {
        "device_placeholder": [],
        "global_placeholder": {},
      }

      p = config.p()
      batch_size, seq_len = None, None
      if p.use_tpu:
        batch_size = p.batch_size if self._is_train else p.eval_batch_size
        seq_len = p.max_len if self._is_train else p.eval_max_len

      # if you need change the data feeding operation;
      #   you also need change the placeholders here
      for shard_id in range(max(self._num_devices, 1)):
        with tf.device("/cpu:0"):
          feature = {
            "source": tf.placeholder(tf.int32, [batch_size, seq_len], "source_%d" % shard_id),
          }
          if p.use_lang_specific_modeling:
            feature["to_lang"] = tf.placeholder(tf.int32, [batch_size], "target_language_%d" % shard_id)
          if self._is_train or self._is_score:
            feature["target"] = tf.placeholder(tf.int32, [batch_size, seq_len], "target_%d" % shard_id)
          placeholders["device_placeholder"].append(feature)

      # Change to TF learning rate schedule, don't need these anymore
      # # learning rate, only for training
      # if self._is_train:
      #   lr = tf.placeholder(tf.float32, [], "learn_rate")
      #   placeholders["global_placeholder"]["lr"] = lr

      return placeholders

    if self._placeholders is None:
      self._placeholders = _get()

    return self._placeholders

  def feed_placeholders(self, shard_datas, global_datas=None):
    """Note that the feeding list must be exactly matched,
    So, if you have fewer data shards for all devices, feed some
    dummy inputs to the model and postprocess these dummies!
    """
    assert self._placeholders is not None, 'please call get_placeholders first'

    feeddict = {}
    for i, shard_data in enumerate(shard_datas):
      feature = self._placeholders["device_placeholder"][i]
      for name in shard_data:
        if name in feature:
          feeddict[feature[name]] = shard_data[name]

    if global_datas is not None:
      feature = self._placeholders['global_placeholder']
      for name in global_datas:
        if name in feature:
          feeddict[feature[name]] = global_datas[name]

    return feeddict
