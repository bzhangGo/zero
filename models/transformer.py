# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

import func
import config
import search
from models import model
from utils import util, dtype
from modules import initializer as tfinit


def encoder(source, p):
  mask = dtype.tf_to_float(tf.cast(source, tf.bool))
  hidden_size = p.hidden_size

  embed_init = tf.random_normal_initializer(0.0, hidden_size ** -0.5)

  embed_name = "embedding" if p.shared_source_target_embedding \
    else "src_embedding"
  src_emb = tf.get_variable(embed_name,
                            [p.src_vocab.size(), p.embed_size],
                            initializer=embed_init)
  src_bias = tf.get_variable("bias", [p.embed_size])

  inputs = func.embedding_layer(
    src_emb, source, one_hot=p.use_tpu) * (hidden_size ** 0.5)
  inputs = tf.nn.bias_add(inputs, src_bias)

  inputs = func.add_timing_signal(inputs)
  inputs = util.valid_apply_dropout(inputs, p.dropout)

  with tf.variable_scope("encoder"):
    x = inputs
    for layer in range(p.num_encoder_layer):
      if p.deep_transformer_init:
        layer_init = tf.variance_scaling_initializer(
          p.initializer_gain * (layer + 1) ** -0.5,
          mode="fan_avg",
          distribution="uniform")
      else:
        layer_init = None
      with tf.variable_scope("layer_{}".format(layer), initializer=layer_init):
        with tf.variable_scope("self_attention"):
          y = func.dot_attention(
            x,
            None,
            func.attention_bias(mask, "masking"),
            hidden_size,
            num_heads=p.num_heads,
            dropout=p.attention_dropout
          )

          y = y['output']
          x = func.residual_fn(x, y, dropout=p.residual_dropout)
          x = func.layer_norm(x)

        with tf.variable_scope("feed_forward"):
          y = func.ffn_layer(
            x,
            p.filter_size,
            hidden_size,
            dropout=p.relu_dropout,
          )

          x = func.residual_fn(x, y, dropout=p.residual_dropout)
          x = func.layer_norm(x)

  source_encodes = x
  x_shp = util.shape_list(x)

  return {
    "encodes": source_encodes,
    "decoder_initializer": {
      "layer_{}".format(l): {
        "k": dtype.tf_to_float(tf.zeros([x_shp[0], p.decode_max_length, hidden_size])),
        "v": dtype.tf_to_float(tf.zeros([x_shp[0], p.decode_max_length, hidden_size])),
      }
      for l in range(p.num_decoder_layer)
    },
    "mask": mask
  }


def decoder(target, state, p):
  mask = dtype.tf_to_float(tf.cast(target, tf.bool))
  hidden_size = p.hidden_size

  embed_init = tf.random_normal_initializer(0.0, hidden_size ** -0.5)
  is_training = p.is_training

  embed_name = "embedding" if p.shared_source_target_embedding \
    else "tgt_embedding"
  tgt_emb = tf.get_variable(embed_name,
                            [p.tgt_vocab.size(), p.embed_size],
                            initializer=embed_init)
  tgt_bias = tf.get_variable("bias", [p.embed_size])

  inputs = func.embedding_layer(
    tgt_emb, target, one_hot=p.use_tpu) * (hidden_size ** 0.5)
  inputs = tf.nn.bias_add(inputs, tgt_bias)

  # shift
  if is_training:
    inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])
    inputs = inputs[:, :-1, :]
    inputs = func.add_timing_signal(inputs)
  else:
    inputs = tf.cond(tf.reduce_all(tf.equal(target, p.tgt_vocab.pad())),
                     lambda: tf.zeros_like(inputs),
                     lambda: inputs)
    # construct constant-compilation time masking
    mask_one_hot = tf.one_hot(
      tf.cast(state['time'], tf.int32),
      depth=p.decode_max_length,
      dtype=inputs.dtype
    )
    mask = tf.cumsum(mask_one_hot, reverse=True)
    mask = util.expand_tile_dims(mask, util.shape_list(inputs)[0], axis=0)

    inputs = func.add_timing_signal(inputs, time=dtype.tf_to_float(state['time']))

  inputs = util.valid_apply_dropout(inputs, p.dropout)

  with tf.variable_scope("decoder"):
    x = inputs
    for layer in range(p.num_decoder_layer):
      if p.deep_transformer_init:
        layer_init = tf.variance_scaling_initializer(
          p.initializer_gain * (layer + 1) ** -0.5,
          mode="fan_avg",
          distribution="uniform")
      else:
        layer_init = None
      with tf.variable_scope("layer_{}".format(layer), initializer=layer_init):
        with tf.variable_scope("self_attention"):
          y = func.dot_attention(
            x,
            None,
            func.attention_bias(tf.shape(mask)[1], "causal") if is_training else
            func.attention_bias(mask, "masking"),
            hidden_size,
            num_heads=p.num_heads,
            dropout=p.attention_dropout,
            decode_step=None if is_training else state['time'],
            cache=None if is_training else
            state['decoder']['state']['layer_{}'.format(layer)]
          )
          if not is_training:
            # k, v
            state['decoder']['state']['layer_{}'.format(layer)] \
              .update(y['cache'])

          y = y['output']
          x = func.residual_fn(x, y, dropout=p.residual_dropout)
          x = func.layer_norm(x)

        with tf.variable_scope("cross_attention"):
          y = func.dot_attention(
            x,
            state['encodes'],
            func.attention_bias(state['mask'], "masking"),
            hidden_size,
            num_heads=p.num_heads,
            dropout=p.attention_dropout,
            decode_step=None if is_training else state['time'],
            cache=None if is_training else
            state['decoder']['state']['layer_{}'.format(layer)]
          )
          if not is_training:
            # mk, mv
            state['decoder']['state']['layer_{}'.format(layer)] \
              .update(y['cache'])

          y = y['output']
          x = func.residual_fn(x, y, dropout=p.residual_dropout)
          x = func.layer_norm(x)

        with tf.variable_scope("feed_forward"):
          y = func.ffn_layer(
            x,
            p.filter_size,
            hidden_size,
            dropout=p.relu_dropout,
          )

          x = func.residual_fn(x, y, dropout=p.residual_dropout)
          x = func.layer_norm(x)
  feature = x

  embed_name = "tgt_embedding" if p.shared_target_softmax_embedding \
    else "softmax_embedding"
  embed_name = "embedding" if p.shared_source_target_embedding \
    else embed_name
  softmax_emb = tf.get_variable(embed_name,
                                [p.tgt_vocab.size(), p.embed_size],
                                initializer=embed_init)
  feature = tf.reshape(feature, [-1, p.embed_size])
  logits = tf.matmul(feature, softmax_emb, False, True)

  logits = tf.cast(logits, tf.float32)

  soft_label, normalizer = util.label_smooth(
    target,
    util.shape_list(logits)[-1],
    factor=p.label_smooth)
  centropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits,
    labels=soft_label
  )
  centropy -= normalizer
  centropy = tf.reshape(centropy, tf.shape(target))

  # for tpu, all 0s padding is possible, fix the mask -> nan issue
  mask = tf.cast(mask, tf.float32)
  per_sample_loss = tf.reduce_sum(centropy * mask, -1) / (tf.reduce_sum(mask, -1) + 1e-8)
  per_sample_mask = tf.cast(tf.cast(tf.reduce_sum(mask, -1), tf.bool), tf.float32)
  loss = tf.reduce_sum(per_sample_loss * per_sample_mask) / (tf.reduce_sum(per_sample_mask) + 1e-8)

  return loss, logits, state, per_sample_loss


def train_fn(features, initializer=None):
  p = config.p()
  p = copy.copy(p)

  # switch the building mode to training one, but notice this is runtime behavior,
  # you must manage it by yourself, and don't use it on other places
  p.is_training = True

  if initializer is None:
    initializer = tfinit.get_initializer(p.initializer, p.initializer_gain)
  with tf.variable_scope(p.scope_name or "model",
                         initializer=initializer,
                         reuse=tf.AUTO_REUSE,
                         dtype=tf.as_dtype(dtype.floatx()),
                         custom_getter=dtype.float32_variable_storage_getter):
    state = encoder(features['source'], p)
    loss, logits, state, _ = decoder(features['target'], state, p)

    return {
      "loss": loss
    }


def score_fn(features, initializer=None):
  p = config.p()
  p = copy.copy(p)

  # switch the building mode to scoring one,
  p.is_training = True

  if initializer is None:
    initializer = tfinit.get_initializer(p.initializer, p.initializer_gain)
    
  p = util.closing_dropout(p)
  p.label_smooth = 0.0
  with tf.variable_scope(p.scope_name or "model",
                         initializer=initializer,
                         reuse=tf.AUTO_REUSE,
                         dtype=tf.as_dtype(dtype.floatx()),
                         custom_getter=dtype.float32_variable_storage_getter):
    state = encoder(features['source'], p)
    _, _, _, scores = decoder(features['target'], state, p)

    return {
      "score": scores
    }


def infer_fn(features):
  p = config.p()
  p = copy.copy(p)

  # switch the building mode to inference one
  p.is_training = False

  p = util.closing_dropout(p)

  def encoding_fn():
    with tf.variable_scope(p.scope_name or "model",
                           reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx()),
                           custom_getter=dtype.float32_variable_storage_getter):
      state = encoder(features["source"], p)
      state["decoder"] = {
        "state": state["decoder_initializer"]
      }
      return state

  def decoding_fn(target, state, time):
    with tf.variable_scope(p.scope_name or "model",
                           reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx()),
                           custom_getter=dtype.float32_variable_storage_getter):
      state['time'] = time
      step_loss, step_logits, step_state, _ = decoder(
        target, state, p)
      del state['time']

      return step_logits, step_state

  beam_outputs = search.beam_search(features, encoding_fn, decoding_fn, p)

  return beam_outputs


# register the model, with a unique name
model.model_register("transformer", train_fn, score_fn, infer_fn)
