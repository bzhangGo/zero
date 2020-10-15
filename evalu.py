# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

from utils import queuer, util, metric


def decode_target_token(id_seq, vocab):
    """Convert sequence ids into tokens"""
    valid_id_seq = []
    for tok_id in id_seq:
        if tok_id == vocab.eos() \
                or tok_id == vocab.pad():
            break
        valid_id_seq.append(tok_id)
    return vocab.to_tokens(valid_id_seq)


def decode_hypothesis(seqs, scores, params, mask=None):
    """Generate decoded sequence from seqs"""
    if mask is None:
        mask = [1.] * len(seqs)

    hypoes = []
    marks = []
    for _seqs, _scores, _m in zip(seqs, scores, mask):
        if _m < 1.: continue

        for seq, score in zip(_seqs, _scores):
            # Temporarily, Use top-1 decoding
            best_seq = seq[0]
            best_score = score[0]

            hypo = decode_target_token(best_seq, params.tgt_vocab)
            mark = best_score

            hypoes.append(hypo)
            marks.append(mark)

    return hypoes, marks


def decoding(session, features, out_seqs, out_scores, dataset, params):
    """Performing decoding with exising information"""
    translations = []
    scores = []
    indices = []

    eval_queue = queuer.EnQueuer(
        dataset.batcher(params.eval_batch_size,
                        buffer_size=params.buffer_size,
                        shuffle=False,
                        train=False),
        lambda x: x,
        worker_processes_num=params.process_num,
        input_queue_size=params.input_queue_size,
        output_queue_size=params.output_queue_size,
    )

    def _predict_one_batch(_data_on_gpu):
        feed_dicts = {}

        _step_indices = []
        for fidx, shard_data in enumerate(_data_on_gpu):
            # define feed_dict
            _feed_dict = {
                features[fidx]["source"]: shard_data['src'],
                features[fidx]["source_mask"]: shard_data['src_mask'],
            }
            feed_dicts.update(_feed_dict)

            # collect data indices
            _step_indices.extend(shard_data['index'])

        # pick up valid outputs
        data_size = len(_data_on_gpu)
        valid_out_seqs = out_seqs[:data_size]
        valid_out_scores = out_scores[:data_size]

        _decode_seqs, _decode_scores = session.run(
            [valid_out_seqs, valid_out_scores], feed_dict=feed_dicts)

        _step_translations, _step_scores = decode_hypothesis(
            _decode_seqs, _decode_scores, params
        )

        return _step_translations, _step_scores, _step_indices

    very_begin_time = time.time()
    data_on_gpu = []
    for bidx, data in enumerate(eval_queue):
        if bidx == 0:
            # remove the data reading time
            very_begin_time = time.time()

        data_on_gpu.append(data)
        # use multiple gpus, and data samples is not enough
        if len(params.gpus) > 0 and len(data_on_gpu) < len(params.gpus):
            continue

        start_time = time.time()
        step_outputs = _predict_one_batch(data_on_gpu)
        data_on_gpu = []

        translations.extend(step_outputs[0])
        scores.extend(step_outputs[1])
        indices.extend(step_outputs[2])

        tf.logging.info(
            "Decoding Batch {} using {:.3f} s, translating {} "
            "sentences using {:.3f} s in total".format(
                bidx, time.time() - start_time,
                len(translations), time.time() - very_begin_time
            )
        )

    if len(data_on_gpu) > 0:

        start_time = time.time()
        step_outputs = _predict_one_batch(data_on_gpu)

        translations.extend(step_outputs[0])
        scores.extend(step_outputs[1])
        indices.extend(step_outputs[2])

        tf.logging.info(
            "Decoding Batch {} using {:.3f} s, translating {} "
            "sentences using {:.3f} s in total".format(
                'final', time.time() - start_time,
                len(translations), time.time() - very_begin_time
            )
        )

    return translations, scores, indices


def scoring(session, features, out_scores, dataset, params):
    """Performing decoding with exising information"""
    scores = []
    indices = []

    eval_queue = queuer.EnQueuer(
        dataset.batcher(params.eval_batch_size,
                        buffer_size=params.buffer_size,
                        shuffle=False,
                        train=False),
        lambda x: x,
        worker_processes_num=params.process_num,
        input_queue_size=params.input_queue_size,
        output_queue_size=params.output_queue_size,
    )

    total_entropy = 0.
    total_tokens = 0.

    def _predict_one_batch(_data_on_gpu):
        feed_dicts = {}

        _step_indices = []
        for fidx, shard_data in enumerate(_data_on_gpu):
            # define feed_dict
            _feed_dict = {
                features[fidx]["source"]: shard_data['src'],
                features[fidx]["target"]: shard_data['tgt'],
            }
            feed_dicts.update(_feed_dict)

            # collect data indices
            _step_indices.extend(shard_data['index'])

        # pick up valid outputs
        data_size = len(_data_on_gpu)
        valid_out_scores = out_scores[:data_size]

        _decode_scores = session.run(
            valid_out_scores, feed_dict=feed_dicts)

        _batch_entropy = sum([s * float((d > 0).sum())
                              for shard_data, shard_scores in zip(_data_on_gpu, _decode_scores)
                              for d, s in zip(shard_data['tgt'], shard_scores.tolist())])
        _batch_tokens = sum([(shard_data['tgt'] > 0).sum() for shard_data in _data_on_gpu])

        _decode_scores = [s for _scores in _decode_scores for s in _scores]

        return _decode_scores, _step_indices, _batch_entropy, _batch_tokens

    very_begin_time = time.time()
    data_on_gpu = []
    for bidx, data in enumerate(eval_queue):
        if bidx == 0:
            # remove the data reading time
            very_begin_time = time.time()

        data_on_gpu.append(data)
        # use multiple gpus, and data samples is not enough
        if len(params.gpus) > 0 and len(data_on_gpu) < len(params.gpus):
            continue

        start_time = time.time()
        step_outputs = _predict_one_batch(data_on_gpu)
        data_on_gpu = []

        scores.extend(step_outputs[0])
        indices.extend(step_outputs[1])

        total_entropy += step_outputs[2]
        total_tokens += step_outputs[3]

        tf.logging.info(
            "Decoding Batch {} using {:.3f} s, translating {} "
            "sentences using {:.3f} s in total".format(
                bidx, time.time() - start_time,
                len(scores), time.time() - very_begin_time
            )
        )

    if len(data_on_gpu) > 0:

        start_time = time.time()
        step_outputs = _predict_one_batch(data_on_gpu)

        scores.extend(step_outputs[0])
        indices.extend(step_outputs[1])

        total_entropy += step_outputs[2]
        total_tokens += step_outputs[3]

        tf.logging.info(
            "Decoding Batch {} using {:.3f} s, translating {} "
            "sentences using {:.3f} s in total".format(
                'final', time.time() - start_time,
                len(scores), time.time() - very_begin_time
            )
        )

    scores = [data[1] for data in
              sorted(zip(indices, scores), key=lambda x: x[0])]

    ppl = np.exp(total_entropy / total_tokens)

    return scores, ppl


def eval_metric(trans, target_file, indices=None):
    """BLEU Evaluate """
    target_valid_files = util.fetch_valid_ref_files(target_file)
    if target_valid_files is None:
        return 0.0

    if indices is not None:
        trans = [data[1] for data in sorted(zip(indices, trans), key=lambda x: x[0])]

    references = []
    for ref_file in target_valid_files:
        cur_refs = tf.gfile.Open(ref_file).readlines()
        cur_refs = [line.strip().split() for line in cur_refs]
        references.append(cur_refs)

    references = list(zip(*references))

    return metric.bleu(trans, references)


def dump_tanslation(tranes, output, indices=None):
    """save translation"""
    if indices is not None:
        tranes = [data[1] for data in
                  sorted(zip(indices, tranes), key=lambda x: x[0])]
    with tf.gfile.Open(output, 'w') as writer:
        for hypo in tranes:
            if isinstance(hypo, list):
                writer.write(' '.join(hypo) + "\n")
            else:
                writer.write(str(hypo) + "\n")
    tf.logging.info("Saving translations into {}".format(output))
