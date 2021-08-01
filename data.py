# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import yaml
import numpy as np
from utils.util import batch_indexer, token_indexer


class Dataset(object):
    def __init__(self, src_file, tgt_file,
                 src_vocab, tgt_vocab, max_len=100,
                 batch_or_token='batch',
                 data_leak_ratio=0.5,
                 N_src=1, N_tgt=1, yaml_file=""):
        self.source = src_file
        self.target = tgt_file
        self.src_vocab = src_vocab         # Note source vocabulary here is meaningless
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len             # the maximum length limit for speech and text (should be separate?)
        self.batch_or_token = batch_or_token
        self.data_leak_ratio = data_leak_ratio

        # number of context and yaml file for recording document information
        # by default, yaml_file is empty which means sentence-level modeling
        self.N_src = N_src
        self.N_tgt = N_tgt
        self.yaml_file = yaml_file
        self.is_document_data = (yaml_file != "")

        self.leak_buffer = []

    # handle documents
    def _document_handler(self, documents):
        for docidx, doc in documents:
            for i in range(len(doc)):
                src = [doc[i-j][0] for j in range(self.N_src) if i-j >= 0][::-1]
                tgt = [doc[i-j][1] for j in range(self.N_tgt) if i-j >= 0][::-1]
                l = len(src)

                src = np.concatenate(src, axis=0)
                # we use <pad> to indicate sentence boundary in the target side
                # this should be avoided (TODO: use a new sentence boundary symbol)
                tgt = ' <pad> '.join(tgt)
                tgt = self.tgt_vocab.to_id(tgt.split()[:self.max_len])

                yield (src, tgt, l, i, docidx)

    def load_sentence_level_data(self, src_reader, tgt_reader):
        h = 0
        while True:
            tgt_line = tgt_reader.readline()

            if tgt_line == "":
                break

            src_line = src_reader["audio_{}".format(h)][()]
            tgt_line = tgt_line.strip()
            h += 1

            if tgt_line == "":
                continue

            yield (
                src_line,
                self.tgt_vocab.to_id(tgt_line.split()[:self.max_len]),
                1,      # one sentence pair
                h-1,    # sentence index in the corpus
                -1,     # we don't care about document idx, so set the docidx to -1
            )

    def load_document_level_data(self, src_reader, tgt_reader, sample_list):
        h = 0
        C_size = 128
        cache = []
        document = []
        docidx = 0
        while True:
            tgt_line = tgt_reader.readline()

            if tgt_line == "":
                break

            src_line = src_reader["audio_{}".format(h)][()]
            tgt_line = tgt_line.strip()
            meta_info = sample_list[h]

            if h > 0 and meta_info['wav'] != sample_list[h-1]['wav']:   # new documents
                cache.append((docidx, document))
                docidx += 1
                document = []

                if len(cache) == C_size:
                    for sample in self._document_handler(cache):
                        yield sample
                    cache = []

            document.append((src_line, tgt_line))

            h += 1

            if tgt_line == "":
                continue

        if len(document) > 0:
            cache.append((docidx, document))
        if len(cache) > 0:
            for sample in self._document_handler(cache):
                yield sample

    # loading dataset
    def load_data(self):
        sources = self.source.strip().split(";")
        targets = self.target.strip().split(";")
        yamlfls = self.yaml_file.strip().split(";")

        # sentence-level input
        if not self.is_document_data:
            for source, target in zip(sources, targets):
                with h5py.File(source, 'r') as src_reader, \
                        open(target, 'r') as tgt_reader:
                    for sample in self.load_sentence_level_data(src_reader, tgt_reader):
                        yield sample
        # document-level input
        else:
            for source, target, yamlfl in zip(sources, targets, yamlfls):
                print("Starting loading YAML alignment file")
                with open(yamlfl, 'r') as yaml_reader:
                    sample_list = yaml.safe_load(yaml_reader)
                print("Finishing Loading.")

                with h5py.File(source, 'r') as src_reader, \
                        open(target, 'r') as tgt_reader:
                    for sample in self.load_document_level_data(src_reader, tgt_reader, sample_list):
                        yield sample

    def to_matrix(self, batch):
        batch_size = len(batch)

        src_lens = [len(sample[1]) for sample in batch]
        tgt_lens = [len(sample[2]) for sample in batch]

        # source dimension, such as 40
        src_dim = batch[0][1].shape[-1]

        src_len = min(self.max_len, max(src_lens))
        tgt_len = min(self.max_len, max(tgt_lens))

        s = np.zeros([batch_size, src_len, src_dim], dtype=np.float32)
        m = np.zeros([batch_size, src_len], dtype=np.float32)
        # padding to be `-1`
        t = np.zeros([batch_size, tgt_len], dtype=np.int32) - 1
        x = []
        y = []
        for eidx, sample in enumerate(batch):
            x.append(sample[0])
            y.append((sample[3], sample[4], sample[5]))
            src_ids, tgt_ids = sample[1], sample[2]

            s[eidx, :min(src_len, len(src_ids))] = src_ids[:src_len]
            t[eidx, :min(tgt_len, len(tgt_ids))] = tgt_ids[:tgt_len]
            m[eidx, :min(src_len, len(src_ids))] = 1.0

        # construct sparse label sequence, for ctc training
        seq_indexes = []
        seq_values = []
        for n, sample in enumerate(batch):
            sequence = sample[2][:tgt_len]

            seq_indexes.extend(zip([n] * len(sequence), range(len(sequence))))
            seq_values.extend(sequence)

        seq_indexes = np.asarray(seq_indexes, dtype=np.int64)
        seq_values = np.asarray(seq_values, dtype=np.int32)
        seq_shape = np.asarray([batch_size, tgt_len], dtype=np.int64)

        return x, s, t, m, (seq_indexes, seq_values, seq_shape), y

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
                x, s, t, m, spar, y = self.to_matrix(batch)
                yield {
                    'src': s,
                    'tgt': t,
                    'src_mask': m,
                    'spar': spar,
                    'index': x,
                    'docindex': y,
                    'raw': batch,
                }

        buffer = self.leak_buffer
        self.leak_buffer = []
        for i, (src_ids, tgt_ids, sl, sentidx, docidx) in enumerate(self.load_data()):
            buffer.append((i, src_ids, tgt_ids, sl, sentidx, docidx))
            if len(buffer) >= buffer_size:
                for data in _handle_buffer(buffer):
                    # check whether the data is tailed
                    batch_size = len(data['raw']) if self.batch_or_token == 'batch' \
                        else max(np.sum(data['tgt'] > 0), np.prod(data['src'].shape[:2]))
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
                    else max(np.sum(data['tgt'] > 0), np.prod(data['src'].shape[:2]))
                if train and batch_size < size * self.data_leak_ratio:
                    self.leak_buffer += data['raw']
                else:
                    yield data
