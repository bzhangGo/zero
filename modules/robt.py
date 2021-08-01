# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time
import numpy as np
import tensorflow as tf

import config
import evalu
import feeder
from utils import util, dtype
from func import embedding_layer

"""
ROBT: Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation
Link: https://arxiv.org/abs/2004.11867

Offer target-language specific layer normalization and linear mapping; Also provide RoBT
, the online back-translation algorithm, to enable massively zero-resource translation
directions.
"""


def lang_mapper(x, to_lang, lang_size, use_tpu=False):
  """Mapping x with language-specific modeling"""
  # to_lang: tensor of [batch size]
  # lang_size: scalar, total language size

  x_shp = util.shape_list(x)
  batch_size, x_size = x_shp[0], x_shp[-1]
  # we tried to relax this mapping by factoring this matrix, but not work
  w_lang = tf.get_variable("lang_mapper", [lang_size, x_size * x_size])

  # extract entries from language projection embeddings
  # based on language information
  w = embedding_layer(w_lang, to_lang, one_hot=use_tpu)
  w = tf.reshape(w, [batch_size, x_size, x_size])

  o = tf.einsum('bsi,bij->bsj', x, w)

  return o


def layer_norm(x, lang, lang_size, use_tpu=False, eps=None, scope=None):
  """Layer normalization layer, with language-specific information"""
  if eps is None:
    eps = dtype.epsilon()
  with tf.variable_scope(scope or "layer_norm",
                         dtype=tf.as_dtype(dtype.floatx())):
    batch_size = util.shape_list(x)[0]
    layer_size = util.shape_list(x)[-1]

    scale = tf.get_variable("scale", [lang_size, layer_size],
                            initializer=tf.ones_initializer())
    offset = tf.get_variable("offset", [lang_size, layer_size],
                             initializer=tf.zeros_initializer())

    scale = tf.reshape(
      embedding_layer(scale, lang, one_hot=use_tpu),
      [batch_size, 1, layer_size]
    )
    offset = tf.reshape(
      embedding_layer(offset, lang, one_hot=use_tpu),
      [batch_size, 1, layer_size]
    )

    mean = tf.reduce_mean(x, -1, keep_dims=True)
    var = tf.reduce_mean((x - mean) ** 2, -1, keep_dims=True)

    return scale * (x - mean) * tf.rsqrt(var + eps) + offset


def get_robt_ops(device_graph):
  """RoBT requires some task specific settings, """
  tf.logging.info("Begin Building RoBT Inferring Graph")
  start_time = time.time()

  # set up infer graph (greedy inference)
  p = config.p()

  # save these parameters for backup
  beam_size = p.beam_size
  decode_length = p.decode_length
  batch_size = p.eval_batch_size
  max_len = p.eval_max_len

  p.beam_size = 1
  p.decode_length = 0
  p.eval_batch_size = p.batch_size
  p.eval_max_len = p.max_len

  eval_feeder = feeder.Feeder(device_graph.num_device(), is_train=False)
  eval_model_op = device_graph.tower_infer_graph(eval_feeder.get_placeholders())

  p.beam_size = beam_size
  p.decode_length = decode_length
  p.eval_batch_size = batch_size
  p.eval_max_len = max_len

  tf.logging.info(
    "End Building RoBT Inferring Graph, within {} seconds".format(time.time() - start_time))
  return eval_feeder, eval_model_op


# random online back-translation
def backtranslate(sess, data_for_shards, eval_feeder, eval_model_op):
  # back-translation procedure
  # sentence pair: (LANG source sentence, target sentence)
  # 0. randomly sample a language: <2lang> (<2lang> != LANG)
  # 1. target sentence => <2lang> target sentence
  # 2. <2lang> target sentence =>MT=> trans' source sentence
  # 3. trans' source sentence => LANG trans' source sentence
  # 4. (LANG trans' source sentence, target sentence)

  p = config.p()

  vocab_to_lang = p.to_lang_vocab
  vocab_src = p.src_vocab
  vocab_tgt = p.tgt_vocab

  def assign_random_lang_id(source, target):
    # prepare random language information to back-translate target sentences
    tgt_lang = source[:, 0]

    fake_lang = []
    fake_to_lang = []
    for lang in tgt_lang:
      select_lang = lang
      select_to_lang = None
      while select_lang == lang:
        # 3 => three special tokens in our vocabulary, useless here
        rand_id = np.random.randint(vocab_to_lang.size() - 3) + 3
        rand_token = vocab_to_lang.get_token(rand_id)
        select_lang = vocab_src.get_id(rand_token)
        select_to_lang = rand_id
      assert select_to_lang is not None
      fake_lang.append(select_lang)
      fake_to_lang.append(select_to_lang)

    fake_source = []
    for lang, tgt in zip(fake_lang, target):
      tgt_tokens = vocab_tgt.to_tokens(list(tgt))
      src_ids = vocab_src.to_id(tgt_tokens, append_eos=False)
      src_sample = [lang] + src_ids
      fake_source.append(src_sample)
    fake_source = np.asarray(fake_source, dtype=np.int32)
    fake_to_lang = np.asarray(fake_to_lang, dtype=np.int32)

    return fake_source, fake_to_lang

  def assign_backtrans(source, target, trans):
    # switch trans->target for back-translation training
    tgt_lang = source[:, 0]
    trans = trans[:, 0, :]

    back_source = []
    for lang, tgt in zip(tgt_lang, trans):
      tgt_tokens = vocab_tgt.to_tokens(list(tgt))
      src_ids = vocab_src.to_id(tgt_tokens, append_eos=False)
      src_sample = [lang] + src_ids
      back_source.append(src_sample)
    back_source = np.asarray(back_source, dtype=np.int32)

    return back_source, target

  # Step 1.1 Prepare [rand lang + `target`] for back-translation
  # data feeding to gpu placeholders
  rlang_data_shards = []
  for fidx, shard_data in enumerate(data_for_shards):
      # sampling random translation direction
      fake_source, fake_to_lang = assign_random_lang_id(shard_data["source"], shard_data["target"])

      rlang_data_shard = copy.deepcopy(shard_data)
      rlang_data_shard.update(
        {
          "source": fake_source,
          "to_lang": fake_to_lang,
        }
      )
      rlang_data_shards.append(rlang_data_shard)

  # Step 1.2 Do online (batch-based) back-translation
  # perform online decoding, greedy with beam-size of 1
  tf.logging.info("Start Online Decoding")
  parsed_decode_outputs = evalu.decode_one_batch(
    sess,
    eval_feeder,
    eval_model_op,
    rlang_data_shards,
    batch_size=p.batch_size if p.use_tpu else int(1e8),
    seq_len=p.max_len if p.use_tpu else int(1e8),
    use_tpu=p.use_tpu,
  )

  decode_seqs = parsed_decode_outputs.aggregate_dictlist_with_key("seq", reduction="list")

  # Step 2.1 Prepare [original lang + `back-translated sentence`] => [original target] for fine-tuning
  # prepare back into the gpu placeholder for back-training
  backtrans_data_shards = []
  for fidx, (shard_data, trans_data) in enumerate(zip(data_for_shards, decode_seqs)):
    # define feed_dict
    source, target = assign_backtrans(shard_data["source"], shard_data["target"], trans_data)

    data_shard = copy.deepcopy(shard_data)

    # length cutting, particularly for TPUs
    source = source[:, :p.max_len]
    target = target[:, :p.max_len]

    if p.use_tpu:
      source = np.concatenate([source, shard_data["source"][source.shape[0]:]], axis=0)
      target = np.concatenate([target, shard_data["target"][target.shape[0]:]], axis=0)

    data_shard.update(
      {
        "source": source,
        "target": target,
      }
    )

    backtrans_data_shards.append(data_shard)

  # Step 2.2 return the new data batch, for parameter tuning
  return backtrans_data_shards
