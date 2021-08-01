# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


class GPUDataFeedingQueue(DeviceDataFeedingQueue):
  """Wrapping independent data feeding, to accelerate GPU training"""
  def __init__(self,
               device_feed,
               queue_size=10000):
    super(GPUDataFeedingQueue, self).__init__(device_feed)

    self._queue_size = queue_size
    self._output_queue = queue.Queue(self._queue_size)

  def wakeup(self):
    pass

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
