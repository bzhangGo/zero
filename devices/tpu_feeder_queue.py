# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import feeder
from utils import util
from devices.device import DeviceDataFeedingQueue

import sys
import time
import threading
import tensorflow as tf

if sys.version[0] == '2':
    import Queue as queue
else:
    import queue as queue


TERMINATION_TOKEN = "<DONE>"


class TPUDataFeedingQueue(DeviceDataFeedingQueue):
  """Wrapping independent data feeding, to accelerate TPU training"""
  def __init__(self,
               device_feed,
               device_graph,
               stop_freq=1000,
               queue_size=10000):
    super(TPUDataFeedingQueue, self).__init__(device_feed)

    self._stop_freq = stop_freq
    self._device_graph = device_graph
    self._enter_sleep = False
    self._queue_size = queue_size

    self._datalist = []
    self._iter_counter = 0
    self._output_queue = queue.Queue(self._queue_size)

    if self._device_feed:
      self._tpu_feeder, self._input_sess, self._infeed_ops = self.get_tpu_feeder()

  def wakeup(self):
    self._enter_sleep = False

  def get_tpu_feeder(self):
    # note the feeder should have separate graph and sessions
    with tf.Graph().as_default():
      # get data feeder
      tpu_feeder = feeder.Feeder(self._device_graph.num_device(), is_train=True)

      # setup session
      cluster_spec = self._device_graph._cluster_resolver.cluster_spec()
      config = tf.ConfigProto(
        isolate_session_state=True,
        allow_soft_placement=True,
      )
      if cluster_spec:
        config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())

      input_sess = tf.Session(
        self._device_graph._cluster_resolver.get_master(),
        config=config,
      )

      # get infeed ops
      infeed_ops = self._device_graph._get_infeed_ops(
        tpu_feeder.get_placeholders()["device_placeholder"])

      return tpu_feeder, input_sess, infeed_ops

  def __iter__(self):
    term_tokens_received = 0
    workers = []

    def worker():
      for lidx, data_chunk in enumerate(self._dataiter):
        if lidx <= self._skip_steps:
          segments = self._skip_steps // 10
          if self._skip_steps < 10 or lidx % segments == 0:
            tf.logging.info(
              "{} Passing {}-th index according to record"
              "".format(util.time_str(time.time()), lidx))
          # skip those data points
          self._output_queue.put(data_chunk)
          continue

        self._datalist.append(data_chunk)

        if self._device_feed and (len(self._datalist) == self._device_graph.num_device()):
          feed_dict = self._tpu_feeder.feed_placeholders(self._datalist)
          self._input_sess.run(self._infeed_ops, feed_dict)
          self._datalist = []
          self._iter_counter += 1
          self._output_queue.put(data_chunk)

          if self._iter_counter > 0 and self._stop_freq > 0 \
            and self._iter_counter % self._stop_freq == 0:
            self._enter_sleep = True
            # waiting for wake up
            while self._enter_sleep:
              time.sleep(1)
        else:
          self._output_queue.put(data_chunk)

      # feeding in termination token
      self._output_queue.put(TERMINATION_TOKEN)

    term_tokens_expected = 1
    workers.append(threading.Thread(target=worker))

    for pr in workers:
      pr.daemon = True
      pr.start()

    while True:
      data_chunk = self._output_queue.get()
      if data_chunk == TERMINATION_TOKEN:
        term_tokens_received += 1
        # need to received all tokens in order to be sure that
        # all data has been processed
        if term_tokens_received == term_tokens_expected:
          for pr in workers:
            pr.join()
          break
        continue
      yield data_chunk
