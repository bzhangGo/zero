# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils.util import batch_indexer, token_indexer


class Dataset(object):
  """This is the dataset reading object, aiming for organizing batches based on
  batch size/token number for training/evaluation.

  Given source (`src_file`) and target (`tgt_file`) source file
  (Format: text corpus, bpe segmented, line by line) with their corresponding
  vocabularies (`src_vocab` and `tgt_vocab`), this class read `buffer_size` instances
  into a internal buffer, and perform sorting, shuffling to this buffer, and then
  extract batches from it.

  Each batch is limited by `size`, either token or batch size, determined by `batch_or_token`.
  In particularly, for TPU running (`use_tpu=True`), only `batch` mode is supported.

  For language-aware modeling, you also need provide the language vocabulary `to_lang_vocab`.

    - During training, we read data iteratively. For the end-of-loading, it's highly possible
      that there are few instances for the batch. We used a `data_leak_ratio`, where batches not
      full-filling `size`(batch/token size)*`data_leak_ratio` will be shifted into a `leak_buffer`.
      Data in this buffer will be reused in the next epoch;

  In case we have fewer instances for training/decoding on TPUs, we add dummy inputs and label
  it as dummy by setting the data index to `-1`.
  """
  def __init__(self,
               src_file,      # source input file
               tgt_file,      # target input file
               src_vocab,     # source vocabulary file
               tgt_vocab,     # target vocabulary file
               size=100,      # token size or batch size
               max_len=100,   # maximum sequence length; when tpu is used, this is the sequence length
               batch_or_token='batch',  # whether use `batch` or `token` based strategy, for tpu, batch only
               data_leak_ratio=0.5,     # for training, if the data doesn't fully fill in the batch, skip it
               use_tpu=False,           # whether use TPU for training/decoding
               to_lang_vocab=None,      # multilingual data, ->xx language vocabulary
               ):
    self.source = src_file
    self.target = tgt_file
    self.src_vocab = src_vocab
    self.tgt_vocab = tgt_vocab
    self.max_len = max_len
    self.batch_or_token = batch_or_token
    self.data_leak_ratio = data_leak_ratio
    self.size = size
    self.use_tpu = use_tpu
    self.to_lang_vocab = to_lang_vocab

    if self.use_tpu:
      assert batch_or_token == 'batch', 'only batch-based data loading supported for TPU devices'

    self.leak_buffer = []

  def load_data(self):
    sources = self.source.strip().split(";")
    targets = self.target.strip().split(";")

    for source, target in zip(sources, targets):
      with tf.gfile.Open(source, 'r') as src_reader, \
        tf.gfile.Open(target, 'r') as tgt_reader:
        while True:
          src_line = src_reader.readline()
          tgt_line = tgt_reader.readline()

          if src_line == "" or tgt_line == "":
            break

          src_line = src_line.strip()
          tgt_line = tgt_line.strip()

          if src_line == "" or tgt_line == "":
            continue

          num_src_words = src_line.count(' ') + 1
          num_tgt_words = tgt_line.count(' ') + 1
          to_lang = "<unk>" if self.to_lang_vocab is None else src_line[:src_line.find(' ')]

          # keep loading part lightweight, improving IO efficiency
          yield src_line, tgt_line, to_lang, num_src_words, num_tgt_words

  def to_matrix(self, batch):
    """Pack inputs into matrices;
    For TPU, we limit the input sequence length as `max_len`;
    """
    batch_size = len(batch)

    # fixing batch size
    if self.use_tpu:
      assert batch_size <= self.size, \
        'Batch is larger than required, {} > {}'.format(batch_size, self.size)
      batch_size = max(batch_size, self.size)

    src_lens = [len(sample[1]) for sample in batch]
    tgt_lens = [len(sample[2]) for sample in batch]

    src_len = min(self.max_len, max(src_lens))
    tgt_len = min(self.max_len, max(tgt_lens))

    # fixing sequence length
    if self.use_tpu:
      src_len = self.max_len
      tgt_len = self.max_len

    s = np.zeros([batch_size, src_len], dtype=np.int32)
    t = np.zeros([batch_size, tgt_len], dtype=np.int32)
    ly = np.zeros([batch_size], dtype=np.int32)

    # a negative index indicates meaningless or dummy padding
    # `x` denotes the index of this sample in the data loading, helping recover
    # sequence order after decoding.
    x = np.ones([batch_size], dtype=np.int32) * -1
    for eidx, sample in enumerate(batch):
      x[eidx] = sample[0]
      ly[eidx] = sample[3]

      src_ids, tgt_ids = sample[1], sample[2]
      s[eidx, :min(src_len, len(src_ids))] = src_ids[:src_len]
      t[eidx, :min(tgt_len, len(tgt_ids))] = tgt_ids[:tgt_len]

    return x, s, t, ly

  def processor(self, token_batch):
    """Convert list instances into numpy inputs;
    Used by processing processor.
    """
    # step 1, convert tokens into ids
    batch = []
    for sample in token_batch:
      src_sample, tgt_sample, to_lang_token = sample[1:4]

      src_ids = self.src_vocab.to_id(src_sample.strip().split()[:self.max_len-1])
      tgt_ids = self.tgt_vocab.to_id(tgt_sample.strip().split()[:self.max_len-1])

      # add to_lang part, which indicates the target language this source input
      # should be translated into.
      # -1: meaningless language id
      to_lang = -1 if self.to_lang_vocab is None else self.to_lang_vocab.get_id(to_lang_token)

      batch.append((sample[0], src_ids, tgt_ids, to_lang))

    # step 2, convert list into numpy matrix
    x, s, t, ly = self.to_matrix(batch)
    features = {
      'source': s,
      'target': t,
      'index': x,
    }

    # add ->xx language information
    if self.to_lang_vocab is not None:
      features['to_lang'] = ly

    # - If tf.data.Dataset could support batch-based generator, that would be great!
    return features

  # a much tricky method for tpu running is to perform
  # data packing, like [sent1 </s> sent2 </s>] for batching
  # attention masks must be modified to accommodate this change
  # for this project, let's keep things simple and naive;
  def batcher(self, buffer_size=1000, shuffle=True, train=True):
    """Reading data and organize into batches (list object);
    Used by reading processor;
    """
    def _handle_buffer(_buffer):
      sorted_buffer = sorted(
        _buffer, key=lambda xx: max(xx[4], xx[5]))

      if self.use_tpu and train:
          np.random.shuffle(sorted_buffer)

      if self.batch_or_token == 'batch':
        buffer_index = batch_indexer(len(sorted_buffer), self.size)
      else:
        buffer_index = token_indexer(
          [[sample[4], sample[5]] for sample in sorted_buffer], self.size)

      index_over_index = batch_indexer(len(buffer_index), 1)
      if shuffle: 
        np.random.shuffle(index_over_index)

      for ioi in index_over_index:
        index = buffer_index[ioi[0]]
        batch = [sorted_buffer[ii] for ii in index]
        yield batch

    def _get_data_size(batch):
      src_lens = [sample[4] for sample in batch]
      tgt_lens = [sample[5] for sample in batch]

      return max(np.sum(src_lens), np.sum(tgt_lens))

    buffer = self.leak_buffer
    self.leak_buffer = []
    for i, (src_sample, tgt_sample, tl_token, n_src, n_tgt) in enumerate(self.load_data()):
      buffer.append((i, src_sample, tgt_sample, tl_token, n_src, n_tgt))
      if len(buffer) >= buffer_size:
        for data in _handle_buffer(buffer):
          # check whether the data is tailed
          batch_size = len(data) if self.batch_or_token == 'batch' else _get_data_size(data)
          # although for TPU devices, the batch_size should be exactly equal to the size
          # we offer the flexibility here to allow training sample leaking
          if batch_size < self.size * self.data_leak_ratio:
            self.leak_buffer += data
          else:
            yield data
        buffer = self.leak_buffer
        self.leak_buffer = []

    # deal with data in the buffer
    if len(buffer) > 0:
      for data in _handle_buffer(buffer):
        # check whether the data is tailed
        batch_size = len(data) if self.batch_or_token == 'batch' else _get_data_size(data)
        if train and batch_size < self.size * self.data_leak_ratio:
          self.leak_buffer += data
        else:
          yield data
