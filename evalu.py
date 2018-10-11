# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import queuer, util, bleu


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


def decoding(session, features,
             out_seqs, out_scores, out_mask, dataset, params):
    """Performing decoding with exising information"""
    translations = []
    scores = []
    indices = []

    batcher = dataset.batcher(params.eval_batch_size,
                              buffer_size=params.buffer_size,
                              shuffle=False)
    eval_queue = queuer.EnQueuer(batcher)
    eval_queue.start(workers=params.nthreads,
                     max_queue_size=params.max_queue_size)

    very_begin_time = time.time()
    for bidx, data in enumerate(eval_queue.get()):
        step_indices, source = data['index'], data['src']

        feed_dict = {
            features["source"]: source
        }

        start_time = time.time()
        decode_seqs, decode_scores, decode_mask = session.run(
            [out_seqs, out_scores, out_mask], feed_dict=feed_dict)

        step_translations, step_scores = decode_hypothesis(
            decode_seqs, decode_scores, params, mask=decode_mask
        )
        translations.extend(step_translations)
        scores.extend(step_scores)
        indices.extend(step_indices)

        tf.logging.info(
            "Decoding Batch {} using {:.3f} s, translating {} "
            "sentences using {:.3f} s in total".format(
                bidx, time.time() - start_time,
                len(translations), time.time() - very_begin_time
            )
        )

    eval_queue.stop()

    return translations, scores, indices


def eval_metric(trans, target_file, indices=None):
    """BLEU Evaluate """
    target_valid_files = util.fetch_valid_ref_files(target_file)
    if target_valid_files is None:
        return 0.0

    if indices is not None:
        trans = [data[1] for data in
                 sorted(zip(indices, trans), key=lambda x:x[0])]

    references = []
    for ref_file in target_valid_files:
        cur_refs = tf.gfile.Open(ref_file).readlines()
        cur_refs = [line.strip().split() for line in cur_refs]
        references.append(cur_refs)

    references = list(zip(*references))

    return bleu.bleu(trans, references)


def dump_tanslation(tranes, output, indices=None):
    """save translation"""
    if indices is not None:
        tranes = [data[1] for data in
                  sorted(zip(indices, tranes), key=lambda x: x[0])]
    with tf.gfile.Open(output, 'w') as writer:
        for hypo in tranes:
            writer.write(' '.join(hypo) + "\n")
    tf.logging.info("Saving translations into {}"
                    .format(output))
