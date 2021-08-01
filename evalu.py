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
        if tok_id < 0:
            continue
        valid_id_seq.append(tok_id)
    return vocab.to_tokens(valid_id_seq)


def swbd_decode_target_token(id_seq, vocab):
    """Convert sequence ids into tokens"""
    valid_id_seq = []
    for tok_id in id_seq:
        if tok_id < 0:
            continue
        valid_id_seq.append(tok_id)

    # removing paddings at the end of generation
    i = len(valid_id_seq) - 1
    while i >= 0 and (valid_id_seq[i] == vocab.pad() or valid_id_seq[i] == vocab.eos()):
        i -= 1
    # only output the last sentence as the translation
    decode_id_seq = []
    while i >= 0 and valid_id_seq[i] != vocab.pad() and valid_id_seq[i] != vocab.eos():
        decode_id_seq.append(valid_id_seq[i])
        i -= 1
    valid_id_seq = decode_id_seq[::-1]

    return vocab.to_tokens(valid_id_seq)


def cbd_decode_target_token(id_seq, vocab):
    """Convert sequence ids into tokens"""
    valid_id_seq = []
    for tok_id in id_seq:
        if tok_id == vocab.eos():
            break
        if tok_id < 0:
            continue
        valid_id_seq.append(tok_id)

    # removing paddings at the end of generation
    i = len(valid_id_seq) - 1
    while i >= 0 and (valid_id_seq[i] == vocab.pad() or valid_id_seq[i] == vocab.eos()):
        i -= 1

    # split translation into multiple segments based on <pad>
    translations = [[]]
    for token in vocab.to_tokens(valid_id_seq[:i+1]):
        if token == "<pad>":
            translations.append([])
        else:
            translations[-1].append(token)
    return translations


def decode_hypothesis(seqs, scores, params, mask=None, mode="default"):
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

            if mode == 'default':
                hypo = decode_target_token(best_seq, params.tgt_vocab)
            elif mode == 'swbd':
                hypo = swbd_decode_target_token(best_seq, params.tgt_vocab)
            elif mode == 'cbd':
                hypo = cbd_decode_target_token(best_seq, params.tgt_vocab)
            else:
                raise NotImplementedError('Inference mode %s is not supported' % mode)
            mark = best_score

            hypoes.append(hypo)
            marks.append(mark)

    return hypoes, marks


def _decoding(session, features, out_seqs, out_scores, dataset, params, mode="default"):
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
            for x, y in zip(shard_data['index'], shard_data['docindex']):
                _step_indices.append((x, y))

        # pick up valid outputs
        data_size = len(_data_on_gpu)
        valid_out_seqs = out_seqs[:data_size]
        valid_out_scores = out_scores[:data_size]

        _decode_seqs, _decode_scores = session.run(
            [valid_out_seqs, valid_out_scores], feed_dict=feed_dicts)

        _step_translations, _step_scores = decode_hypothesis(
            _decode_seqs, _decode_scores, params, mode=mode
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


def default_decoding(session, features, out_seqs, out_scores, dataset, params):
    """Orignal decoding method."""
    tf.logging.info('Default Decoding is activated!')
    translations, scores, indices = _decoding(session, features, out_seqs, out_scores, dataset, params, mode='default')
    indices = [idx[0] for idx in indices]   # dropping the docindex
    return translations, scores, indices


def swbd_decoding(session, features, out_seqs, out_scores, dataset, params):
    """Sliding Window-Based Contextual Inference.

    Given input x1, x2, x3, the model will generate y1, y2 and y3. Then only y3 is retained as the translation of x3.
    Other translations y1 and y2 are discarded.

    After than, the window moves to the next, forming input x2, x3, x4, which will be translated into y2, y3 and y4.
    Again, only y4 is preserved for translation of x4.

    In this decoding approach, the target-side context is locally consistent, but not globally consistent.
    """
    tf.logging.info('SWBD Decoding is activated!')
    translations, scores, indices = _decoding(session, features, out_seqs, out_scores, dataset, params, mode='swbd')
    indices = [idx[0] for idx in indices]   # dropping the docindex
    return translations, scores, indices


def cbd_decoding(session, features, out_seqs, out_scores, dataset, params):
    """Chunk-based Contextual Inference.

    Given input x1, x2, x3, the model will generate y1, y2 and y3, and [y1,y2,y3] will be regarded as the translation.

    Next the model will translate x4, x5, x6, i.e. one chunk after another

    Again, this method is also locally consistent.
    """
    tf.logging.info('CBD Decoding is activated!')
    assert params.N_src == params.N_tgt, 'We assume that the source and target have the same number of sequences in cbd'
    translations, scores, indices = _decoding(session, features, out_seqs, out_scores, dataset, params, mode='cbd')

    # x[0][1] => docindex => (sl, sentidx, docidx)
    sorted_outputs = sorted(zip(indices, translations, scores), key=lambda x: (x[0][1][2], x[0][1][1]))

    documents = [[]]
    docidx = 0
    for idx, trans, score in sorted_outputs:
        trans_docidx = idx[1][2]
        if trans_docidx != docidx:
            docidx = trans_docidx
            documents.append([])
        documents[-1].append((idx, trans, score))

    new_translations, new_scores = [], []
    for document in documents:
        doc_length = len(document)
        N_src = params.N_src

        i = doc_length - 1
        doc_translations = []
        doc_scores = []
        while i >= 0:
            segment = document[i]
            num_sentence = segment[0][1][0]
            translation = segment[1]

            assert num_sentence == N_src or num_sentence == (i+1)

            local_scores, local_translations = [], []
            for j in range(num_sentence):
                if j < len(translation):
                    local_scores.append(segment[2])
                    local_translations.append(translation[j])
                else:
                    local_scores.append(-1.0)
                    local_translations.append("")     # misaligned translation

            doc_translations.extend(local_translations[::-1])
            doc_scores.extend(local_scores[::-1])

            i -= N_src
        new_translations.extend(doc_translations[::-1])
        new_scores.extend(doc_scores[::-1])

    translations = new_translations
    scores = new_scores
    indices = list(range(len(translations)))
    assert len(translations) == len(sorted_outputs)

    return translations, scores, indices


def imed_decoding(session, features, out_seqs, out_scores, dataset, params):
    """In-model ensemble decoding.

    Given x1, x2, x3, and previous translation y1', y2', we obtain y3' as follows:
        p(y3'| x1, x2, x3, y1', y2') (i.e. document-level ST) + \lambda p(y3'|x3)
    Both document-level decoding and sentence-level decoding are ensembled together for the translation

    Note we use the same document-level ST model for both inference, thus we call it `in-model`.
    """
    tf.logging.info('IMED Decoding is activated!')
    translations = []
    scores = []
    indices = []

    # only allow sentence-level input
    dataset.N_src = 1
    dataset.N_tgt = 1
    eval_queue = queuer.EnQueuer(
        dataset.batcher(1,  # params.eval_batch_size,
                        buffer_size=params.buffer_size,
                        shuffle=False,
                        train=False),
        lambda x: x,
        worker_processes_num=params.process_num,
        input_queue_size=params.input_queue_size,
        output_queue_size=params.output_queue_size,
    )

    # collecting documents
    corpus = []
    for bidx, data in enumerate(eval_queue):
        corpus.append(data)
    sorted_corpus = sorted(corpus, key=lambda x: (x['docindex'][0][2], x['docindex'][0][1]))

    docidx = 0
    documents = [[]]
    for data in sorted_corpus:
        sample_docidx = data['docindex'][0][2]

        if docidx != sample_docidx:
            documents.append([])
            docidx = sample_docidx

        documents[-1].append(data)

    def _predict_one_batch(_data_on_gpu):
        feed_dicts = {}

        _step_indices = []
        for fidx, shard_data in enumerate(_data_on_gpu):
            # define feed_dict
            _feed_dict = {
                features[fidx]["source"]: shard_data['src'],
                features[fidx]["target"]: shard_data['tgt'],
                features[fidx]["source_mask"]: shard_data['src_mask'],
                features[fidx]["sent_source"]: shard_data['sent_src'],
                features[fidx]["sent_source_mask"]: shard_data['sent_src_mask'],
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
            _decode_seqs, _decode_scores, params,
        )

        return _step_translations, _step_scores, _step_indices

    very_begin_time = time.time()
    assert len(params.gpus) == 1, 'only support one gpu decoding in this case'

    pad_id = params.tgt_vocab.pad()

    def _predict_one_document(doc, counter):
        tgts = []

        hypos = []
        scores = []
        indices = []
        for i in range(len(doc)):
            src = [v['raw'][0][1] for v in doc[max(i - params.N_src + 1, 0):i + 1]]
            src = np.concatenate(src, axis=0)

            sent_src = doc[i]['raw'][0][1]

            tgt = tgts[-(params.N_tgt - 1):]

            batch_src = src[None, :, :]
            batch_sent_src = sent_src[None, :, :]
            batch_tgt = [[pad_id]]
            batch_mask = np.ones([1, src.shape[0]], dtype=np.float32)
            batch_sent_mask = np.ones([1, sent_src.shape[0]], dtype=np.float32)

            for t in tgt:
                batch_tgt[0].extend(t)
                batch_tgt[0].append(pad_id)

            cur_batch = {
                "src": batch_src,
                "tgt": batch_tgt,
                "src_mask": batch_mask,
                "index": [counter],
                "sent_src": batch_sent_src,
                "sent_src_mask": batch_sent_mask,
            }
            counter += 1

            step_outputs = _predict_one_batch([cur_batch])

            hypos.extend(step_outputs[0])
            scores.extend(step_outputs[1])
            indices.extend(step_outputs[2])

            assert len(step_outputs[0]) == 1
            hypo = step_outputs[0][0]
            hypo = params.tgt_vocab.to_id(hypo, append_eos=False)

            tgts.append(hypo)

        return hypos, scores, indices

    for bidx, doc in enumerate(documents):
        if bidx == 0:
            # remove the data reading time
            very_begin_time = time.time()

        start_time = time.time()
        step_outputs = _predict_one_document(doc, len(scores))

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

    return translations, scores, indices


def decoding(session, features, out_seqs, out_scores, dataset, params):
    # sentence-level translation
    if params.inference_mode == "default":
        return default_decoding(session, features, out_seqs, out_scores, dataset, params)
    elif params.inference_mode == "swbd":
        return swbd_decoding(session, features, out_seqs, out_scores, dataset, params)
    elif params.inference_mode == "swbd_cons":
        return imed_decoding(session, features, out_seqs, out_scores, dataset, params)
    elif params.inference_mode == "imed":
        return imed_decoding(session, features, out_seqs, out_scores, dataset, params)
    elif params.inference_mode == "cbd":
        return cbd_decoding(session, features, out_seqs, out_scores, dataset, params)
    else:
        raise NotImplementedError('Inference with %s is not supported' % params.inference_mode)


def scoring(session, features, out_scores, out_memory, dataset, params):
    """Performing decoding with exising information"""
    scores = []
    indices = []
    memory = []

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
                features[fidx]["source_mask"]: shard_data['src_mask'],
            }
            feed_dicts.update(_feed_dict)

            # collect data indices
            _step_indices.extend(shard_data['index'])

        # pick up valid outputs
        data_size = len(_data_on_gpu)
        valid_out_scores = out_scores[:data_size]
        valid_out_memory = out_memory[:data_size]

        _decode_scores, _decode_memory = session.run(
            [valid_out_scores, valid_out_memory], feed_dict=feed_dicts)

        _batch_entropy = sum([s * float((d > 0).sum())
                              for shard_data, shard_scores in zip(_data_on_gpu, _decode_scores)
                              for d, s in zip(shard_data['tgt'], shard_scores.tolist())])
        _batch_tokens = sum([(shard_data['tgt'] > 0).sum() for shard_data in _data_on_gpu])

        _decode_scores = [s for _scores in _decode_scores for s in _scores]
        _decode_memory = [s for _memory in _decode_memory for s in _memory]

        return _decode_scores, _step_indices, _batch_entropy, _batch_tokens, _decode_memory

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
        memory.extend(step_outputs[4])

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
        memory.extend(step_outputs[4])

        tf.logging.info(
            "Decoding Batch {} using {:.3f} s, translating {} "
            "sentences using {:.3f} s in total".format(
                'final', time.time() - start_time,
                len(scores), time.time() - very_begin_time
            )
        )

    scores = [data[1] for data in
              sorted(zip(indices, scores), key=lambda x: x[0])]
    memory = [data[1] for data in
              sorted(zip(indices, memory), key=lambda x: x[0])]

    ppl = np.exp(total_entropy / total_tokens)

    return scores, ppl, memory


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
