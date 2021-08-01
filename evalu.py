# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import copy
import numpy as np
import tensorflow as tf

from devices import device
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


def decode_hypothesis(parsed_outputs, p):
  """Generate decoded sequence from seqs"""

  hypoes = []
  marks = []
  for device_output in parsed_outputs.get_output():

    seqs = device_output["seq"]
    scores = device_output["score"]

    for seq, score in zip(seqs, scores):
      # Use top-1, or best decoding
      best_seq = seq[0]
      best_score = score[0]

      hypo = decode_target_token(best_seq, p.tgt_vocab)
      mark = best_score

      hypoes.append(hypo)
      marks.append(mark)

  return hypoes, marks


def _padding_data_batches(data_points, num_device):
  """padding the data so that all devices are used"""
  if len(data_points) == num_device:
    return data_points

  sample = copy.deepcopy(data_points[-1])
  sample["index"] = sample["index"] * 0 - 1

  pad_num = num_device - len(data_points)
  for _ in range(pad_num):
    data_points.append(sample)
  return data_points


def decode_one_batch(sess,            # running session
                     eval_feeder,     # data feeder
                     eval_model_op,   # modeling operation, actual running graph
                     data_for_shards, # data on tpus/gpus/cpus
                     batch_size,      # maximum allowed batch size
                     seq_len,         # maximum allowed sequence length, both consider TPU
                     use_tpu=False,   # whether use tpu
                     ):

  # data feeding to gpu placeholders
  feed_dicts = eval_feeder.feed_placeholders(data_for_shards)
  num_device = len(eval_feeder.get_placeholders()['device_placeholder'])

  # padding if necessary
  def _maybe_cut_or_padding(y, pad=False):
    pattern = []
    for _ in range(y.ndim):
      pattern.append([0, 0])
    y = y[:batch_size]
    pattern[0][1] = batch_size - y.shape[0]

    if y.ndim > 1:
      y = y[:, :seq_len]
      pattern[1][1] = seq_len - y.shape[1]

    # padding zeros if necessary
    if not pad:
      return y
    else:
      return np.pad(y, pattern, 'constant')

  def _maybe_cut_outputs(parsed_outputs, n_samples):
    outputs = parsed_outputs.get_output()

    if n_samples < batch_size:
      for output in outputs:
        for k in output:
          if np.asarray(output[k]).ndim > 0:
            output[k] = output[k][:n_samples]

    return parsed_outputs

  # adjust feature batch size, to fit eval_batch_size
  n_samples = None
  for placeholder in feed_dicts:
    x = feed_dicts[placeholder]
    if np.asarray(x).ndim < 1:
      # in case of scalar value
      feed_dicts[placeholder] = x
    else:
      # assume tensors of shape [batch_size, seq_len, ...]
      n_samples = x.shape[0]
      feed_dicts[placeholder] = _maybe_cut_or_padding(x, pad=use_tpu)

  _, _, decode_outputs = sess.run(
    [eval_model_op.infeed_op, eval_model_op.execute_op, eval_model_op.outfeed_op],
    feed_dict=feed_dicts,
  )
  parsed_decode_outputs = eval_model_op.outfeed_mapper.parse(
    decode_outputs, max(num_device, 1)
  )
  if use_tpu:
    parsed_decode_outputs = _maybe_cut_outputs(parsed_decode_outputs, n_samples)

  return parsed_decode_outputs


def decoding(session, eval_feeder, eval_model_op, dataset, p):
  """Performing decoding with existing information"""
  translations = []
  scores = []
  indices = []

  num_device = len(eval_feeder.get_placeholders()['device_placeholder'])

  eval_queue = queuer.EnQueuer(
    dataset.batcher(buffer_size=p.buffer_size,
                    shuffle=False,
                    train=False),
    lambda x: dataset.processor(x),
    worker_processes_num=p.process_num,
    input_queue_size=p.input_queue_size,
    output_queue_size=p.output_queue_size,
  )

  def _predict_one_batch(_data_for_shards):
    _step_indices = []
    for fidx, shard_data in enumerate(_data_for_shards):
      # collect data indices
      _step_indices.extend(shard_data['index'])

    _parsed_decode_outputs = decode_one_batch(
      session,
      eval_feeder,
      eval_model_op,
      _data_for_shards,
      batch_size=p.eval_batch_size if p.use_tpu else int(1e8),
      seq_len=p.eval_max_len if p.use_tpu else int(1e8),
      use_tpu=p.use_tpu,
    )

    # id to tokens
    _step_translations, _step_scores = decode_hypothesis(
      _parsed_decode_outputs, p
    )

    return _step_translations, _step_scores, _step_indices

  very_begin_time = time.time()
  data_for_shards = []
  for bidx, data in enumerate(eval_queue):
    if bidx == 0:
      # remove the data reading time
      very_begin_time = time.time()

    data_for_shards.append(data)
    # use multiple gpus, and data samples is not enough
    if len(data_for_shards) < num_device:
      continue

    start_time = time.time()
    data_for_shards = _padding_data_batches(data_for_shards, num_device)
    step_outputs = _predict_one_batch(data_for_shards)
    data_for_shards = []

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

  if len(data_for_shards) > 0:

    start_time = time.time()

    data_for_shards = _padding_data_batches(data_for_shards, num_device)
    step_outputs = _predict_one_batch(data_for_shards)

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

  filtered_sorted_results = [
    (data[1], data[2])
    for data in sorted(zip(indices, translations, scores), key=lambda x: x[0])
    if data[0] >= 0
  ]

  translations, scores = list(zip(*filtered_sorted_results))

  return translations, scores


def scoring(session, score_feeder, score_model_op, dataset, p):
  """Performing decoding with exising information"""
  scores = {"src": [], "tgt": [], "src_str": [], "tgt_str": []}
  indices = []

  num_device = len(score_feeder.get_placeholders()['device_placeholder'])

  eval_queue = queuer.EnQueuer(
    dataset.batcher(buffer_size=p.buffer_size,
                    shuffle=False,
                    train=False),
    lambda x: dataset.processor(x),
    worker_processes_num=p.process_num,
    input_queue_size=p.input_queue_size,
    output_queue_size=p.output_queue_size,
  )

  total_entropy = 0.
  total_tokens = 0.
  for score_key in score_model_op.outfeed_mapper.get_keys():
      scores[score_key] = []

  def _predict_one_batch(_data_for_shards):
    feed_dicts = score_feeder.feed_placeholders(_data_for_shards)

    _step_indices = []
    for fidx, shard_data in enumerate(_data_for_shards):
      # collect data indices
      _step_indices.extend(shard_data['index'])

    # perform scoreing
    _, _, _decode_outputs = session.run(
      [score_model_op.infeed_op,
       score_model_op.execute_op,
       score_model_op.outfeed_op], feed_dict=feed_dicts)

    # [{seq:[batchxseqs], score:[batchxscores]}] * num_device
    _parsed_decode_outputs = score_model_op.outfeed_mapper.parse(
      _decode_outputs, num_device
    )

    _decode_scores = {
      score_key: [
        s
        for parsed_output in _parsed_decode_outputs.get_output()
        for s in parsed_output[score_key]
      ]
      for score_key in score_model_op.outfeed_mapper.get_keys()
    }

    _target_tokens = [
      float((d > 0).sum())
      for shard_data in _data_for_shards
      for d in shard_data['target']
    ]

    _batch_entropy, _batch_tokens = 0.0, 0
    for s, t, i in zip(_decode_scores["score"], _target_tokens, _step_indices):
      # dummy/padding sample
      if i < 0:
        continue
      _batch_entropy += s * t
      _batch_tokens += t

    return _decode_scores, _step_indices, _batch_entropy, _batch_tokens

  very_begin_time = time.time()
  data_for_shards = []
  for bidx, data in enumerate(eval_queue):
    if bidx == 0:
      # remove the data reading time
      very_begin_time = time.time()

    data_for_shards.append(data)
    # use multiple gpus, and data samples is not enough
    if len(data_for_shards) < num_device:
      continue

    start_time = time.time()
    data_for_shards = _padding_data_batches(data_for_shards, num_device)
    step_outputs = _predict_one_batch(data_for_shards)

    step_scores = step_outputs[0]
    for score_key in step_scores:
      scores[score_key].extend(step_scores[score_key])

    for data_for_shard in data_for_shards:
      scores["src"].extend([v for v in data_for_shard['source']])
      scores["tgt"].extend([v for v in data_for_shard['target']])

      scores["src_str"].extend(
        [decode_target_token(list(v), p.src_vocab) for v in data_for_shard['source']])
      scores["tgt_str"].extend(
        [decode_target_token(list(v), p.tgt_vocab) for v in data_for_shard['target']])

    indices.extend(step_outputs[1])
    data_for_shards = []

    total_entropy += step_outputs[2]
    total_tokens += step_outputs[3]

    tf.logging.info(
      "Decoding Batch {} using {:.3f} s, translating {} "
      "sentences using {:.3f} s in total".format(
        bidx, time.time() - start_time,
        len(scores), time.time() - very_begin_time
      )
    )

  if len(data_for_shards) > 0:

    start_time = time.time()
    data_for_shards = _padding_data_batches(data_for_shards, num_device)
    step_outputs = _predict_one_batch(data_for_shards)

    step_scores = step_outputs[0]
    for score_key in step_scores:
      scores[score_key].extend(step_scores[score_key])

    for data_for_shard in data_for_shards:
      scores["src"].extend([v for v in data_for_shard['source']])
      scores["tgt"].extend([v for v in data_for_shard['target']])

      scores["src_str"].extend(
        [decode_target_token(list(v), p.src_vocab) for v in data_for_shard['source']])
      scores["tgt_str"].extend(
        [decode_target_token(list(v), p.tgt_vocab) for v in data_for_shard['target']])

    indices.extend(step_outputs[1])

    total_entropy += step_outputs[2]
    total_tokens += step_outputs[3]

    tf.logging.info(
      "Decoding Batch {} using {:.3f} s, translating {} "
      "sentences using {:.3f} s in total".format(
        'final', time.time() - start_time,
        len(indices), time.time() - very_begin_time
      )
    )

  for score_key in scores:
    scores[score_key] = [
      data[1]
      for data in sorted(zip(indices, scores[score_key]), key=lambda x: x[0])
      if data[0] >= 0
    ]

  ppl = np.exp(total_entropy / total_tokens)

  return scores, ppl


def eval_metric(trans, target_file):
  """BLEU Evaluate """
  target_valid_files = util.fetch_valid_ref_files(target_file)
  if target_valid_files is None:
    return 0.0

  references = []
  for ref_file in target_valid_files:
    cur_refs = tf.gfile.Open(ref_file).readlines()
    cur_refs = [line.strip().split() for line in cur_refs]
    references.append(cur_refs)

  references = list(zip(*references))

  return metric.bleu(trans, references)


def dump_tanslation(tranes, output):
  """save translation"""
  with tf.gfile.Open(output, 'w') as writer:
    for hypo in tranes:
      if isinstance(hypo, list):
        writer.write(' '.join(hypo) + "\n")
      else:
        writer.write(str(hypo) + "\n")
  tf.logging.info("Saving translations into {}".format(output))
