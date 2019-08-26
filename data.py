# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from utils.util import batch_indexer, token_indexer


class Dataset(object):
    def __init__(self, src_file, tgt_file,
                 src_vocab, tgt_vocab, max_len=100,
                 batch_or_token='batch',
                 data_leak_ratio=0.5):
        self.source = src_file
        self.target = tgt_file
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.batch_or_token = batch_or_token
        self.data_leak_ratio = data_leak_ratio

        self.leak_buffer = []

    def load_data(self):
        with open(self.source, 'r') as src_reader, \
                open(self.target, 'r') as tgt_reader:
            while True:
                src_line = src_reader.readline()
                tgt_line = tgt_reader.readline()

                if src_line == "" or tgt_line == "":
                    break

                src_line = src_line.strip()
                tgt_line = tgt_line.strip()

                if src_line == "" or tgt_line == "":
                    continue

                yield (
                    self.src_vocab.to_id(src_line.strip().split()[:self.max_len]),
                    self.tgt_vocab.to_id(tgt_line.strip().split()[:self.max_len])
                )

    def to_matrix(self, batch):
        batch_size = len(batch)

        src_lens = [len(sample[1]) for sample in batch]
        tgt_lens = [len(sample[2]) for sample in batch]

        src_len = min(self.max_len, max(src_lens))
        tgt_len = min(self.max_len, max(tgt_lens))

        s = np.zeros([batch_size, src_len], dtype=np.int32)
        t = np.zeros([batch_size, tgt_len], dtype=np.int32)
        x = []
        for eidx, sample in enumerate(batch):
            x.append(sample[0])
            src_ids, tgt_ids = sample[1], sample[2]

            s[eidx, :min(src_len, len(src_ids))] = src_ids[:src_len]
            t[eidx, :min(tgt_len, len(tgt_ids))] = tgt_ids[:tgt_len]
        return x, s, t

    def batcher(self, size, buffer_size=1000, shuffle=True, train=True):
        def _handle_buffer(_buffer):
            sorted_buffer = sorted(
                _buffer, key=lambda xx: max(len(xx[1]), len(xx[2])))

            if self.batch_or_token == 'batch':
                buffer_index = batch_indexer(len(sorted_buffer), size)
            else:
                buffer_index = token_indexer(
                    [[len(sample[1]), len(sample[2])] for sample in sorted_buffer], size)

            index_over_index = batch_indexer(len(buffer_index), 1)
            if shuffle: np.random.shuffle(index_over_index)

            for ioi in index_over_index:
                index = buffer_index[ioi[0]]
                batch = [sorted_buffer[ii] for ii in index]
                x, s, t = self.to_matrix(batch)
                yield {
                    'src': s,
                    'tgt': t,
                    'index': x,
                    'raw': batch,
                }

        buffer = self.leak_buffer
        self.leak_buffer = []
        for i, (src_ids, tgt_ids) in enumerate(self.load_data()):
            buffer.append((i, src_ids, tgt_ids))
            if len(buffer) >= buffer_size:
                for data in _handle_buffer(buffer):
                    # check whether the data is tailed
                    batch_size = len(data['raw']) if self.batch_or_token == 'batch' \
                        else max(np.sum(data['tgt'] > 0), np.sum(data['src'] > 0))
                    if batch_size < size * self.data_leak_ratio:
                        self.leak_buffer += data['raw']
                    else:
                        yield data
                buffer = self.leak_buffer
                self.leak_buffer = []

        # deal with data in the buffer
        if len(buffer) > 0:
            for data in _handle_buffer(buffer):
                # check whether the data is tailed
                batch_size = len(data['raw']) if self.batch_or_token == 'batch' \
                    else max(np.sum(data['tgt'] > 0), np.sum(data['src'] > 0))
                if train and batch_size < size * self.data_leak_ratio:
                    self.leak_buffer += data['raw']
                else:
                    yield data
