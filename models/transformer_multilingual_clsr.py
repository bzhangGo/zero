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
from modules import initializer as tfinit, cct


"""
Source code for ICLR Submission: 
SHARE OR NOT?
LEARNING TO SCHEDULE LANGUAGE-SPECIFIC CAPACITY FOR MULTILINGUAL TRANSLATION
https://openreview.net/pdf?id=Wj4ODo0uyCF

We employ conditional computation to analysis when and whether language-specific 
capacity matters for multilingual translation. We add CCT-based language specific
mapping to each sublayer in Transformer, and let the model learn which ones are 
crucial so as to offer us insights about language specific computation.
"""


def _cct_budget(gates, mask):
  # gates: dictionary, contains every kinds of `p_c` matrix
  # mask: the mask upon each batch/token

  bgsum, bgall = 0., 0.
  mask = tf.expand_dims(mask, -1)

  for key in gates:
    value = gates[key]

    bgsum += tf.reduce_sum(value * mask)
    bgall += tf.reduce_sum(tf.ones_like(value) * mask)

  return bgsum, bgall


def _lang_mapper(x, to_lang, w_lang, use_tpu=False):
  """Mapping x with language-specific modeling"""
  # to_lang: tensor of [batch size]
  x_shp = util.shape_list(x)
  batch_size, x_size = x_shp[0], x_shp[-1]

  # extract entries from language projection embeddings
  # based on language information
  w = func.embedding_layer(w_lang, to_lang, one_hot=use_tpu)
  w = tf.reshape(w, [batch_size, x_size, x_size])

  o = tf.einsum('bsi,bij->bsj', x, w)

  return o


def _clsr(x, y, to_lang, w_lang, hidden_size, use_tpu, gater):
  """conditional language-specific routing"""
  # adding share/private layer
  p_y = _lang_mapper(y, to_lang, w_lang, use_tpu=use_tpu)
  s_y = func.linear(y, hidden_size, scope="o_mapper")

  g = gater.gating_layer(x, 1)

  # note: g == 1 corresponding to LS computation
  y = p_y * g + s_y * (1. - g)

  return y, g


def encoder(source, to_lang, p, gater):
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

  gates = {}
  with tf.variable_scope("encoder"):
    x = inputs

    # language specific linear mapping parameters: we share this weight across different sublayers
    w_lang = tf.get_variable("lang_mapper", [p.to_lang_vocab.size(), hidden_size * hidden_size])

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
            dropout=p.attention_dropout,
            out_map=False,
          )

          y = y['output']

          # clsr
          y, g = _clsr(x, y, to_lang, w_lang, hidden_size, p.use_tpu, gater)
          gates['enc_san_%s' % layer] = g

          x = func.residual_fn(x, y, dropout=p.residual_dropout)
          x = func.layer_norm(x)

        with tf.variable_scope("feed_forward"):
          y = func.ffn_layer(
            x,
            p.filter_size,
            hidden_size,
            dropout=p.relu_dropout,
          )

          # clsr
          y, g = _clsr(x, y, to_lang, w_lang, hidden_size, p.use_tpu, gater)
          gates['enc_ffn_%s' % layer] = g

          x = func.residual_fn(x, y, dropout=p.residual_dropout)
          x = func.layer_norm(x)

  source_encodes = x
  x_shp = util.shape_list(x)

  return {
    "encodes": source_encodes,
    "decoder_initializer": {
      "layer_{}".format(l): {
        # for fusion
        "aan": dtype.tf_to_float(tf.zeros([x_shp[0], 1, hidden_size])),
        # for transformer
        "k": dtype.tf_to_float(tf.zeros([x_shp[0], p.decode_max_length, hidden_size])),
        "v": dtype.tf_to_float(tf.zeros([x_shp[0], p.decode_max_length, hidden_size])),
      }
      for l in range(p.num_decoder_layer)
    },
    "mask": mask,
    "to_lang": to_lang,
    "gate": gates,
  }


def decoder(target, state, p, gater):
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

  gates = {}
  with tf.variable_scope("decoder"):
    x = inputs
    to_lang = state["to_lang"]

    # language specific linear mapping parameters: we share this weight across different sublayers
    # maybe also share it with the encoder?
    w_lang = tf.get_variable("lang_mapper", [p.to_lang_vocab.size(), hidden_size * hidden_size])

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
            state['decoder']['state']['layer_{}'.format(layer)],
            out_map=False,
          )
          if not is_training:
            # k, v
            state['decoder']['state']['layer_{}'.format(layer)] \
              .update(y['cache'])

          y = y['output']

          # clsr
          y, g = _clsr(x, y, to_lang, w_lang, hidden_size, p.use_tpu, gater)
          gates['dec_san_%s' % layer] = g

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
            state['decoder']['state']['layer_{}'.format(layer)],
            out_map=False,
          )
          if not is_training:
            # mk, mv
            state['decoder']['state']['layer_{}'.format(layer)] \
              .update(y['cache'])

          y = y['output']

          # clsr
          y, g = _clsr(x, y, to_lang, w_lang, hidden_size, p.use_tpu, gater)
          gates['dec_can_%s' % layer] = g

          x = func.residual_fn(x, y, dropout=p.residual_dropout)
          x = func.layer_norm(x)

        with tf.variable_scope("feed_forward"):
          y = func.ffn_layer(
            x,
            p.filter_size,
            hidden_size,
            dropout=p.relu_dropout,
          )

          # clsr
          y, g = _clsr(x, y, to_lang, w_lang, hidden_size, p.use_tpu, gater)
          gates['dec_ffn_%s' % layer] = g

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

  # conditional bucket loss
  if is_training:
    enc_budget, enc_all_budget = _cct_budget(state['gate'], state['mask'])
    dec_budget, dec_all_budget = _cct_budget(gates, mask)

    used_budget = enc_budget + dec_budget
    total_budget = enc_all_budget + dec_all_budget

    loss += tf.abs(used_budget / (total_budget + 1e-8) - p.cct_bucket_p)

  return loss, logits, state, per_sample_loss, (state['gate'], gates)


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
    gater = cct.CCTGater(p.cct_alpha_value, p.max_training_steps, True, p.cct_relu_dim)

    state = encoder(features['source'], features['to_lang'], p, gater)
    loss, logits, state, _, _ = decoder(features['target'], state, p, gater)

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
    gater = cct.CCTGater(p.cct_alpha_value, p.max_training_steps, False, p.cct_relu_dim)

    state = encoder(features['source'], features['to_lang'], p, gater)
    _, _, _, scores, (enc_gates, dec_gates) = decoder(features['target'], state, p, gater)

    score_results = {"score": scores}
    score_results.update(enc_gates)
    score_results.update(dec_gates)

    return score_results


def infer_fn(features):
  p = config.p()
  p = copy.copy(p)

  # switch the building mode to inference one
  p.is_training = False

  p = util.closing_dropout(p)

  gater = cct.CCTGater(p.cct_alpha_value, p.max_training_steps, False, p.cct_relu_dim)

  def encoding_fn():
    with tf.variable_scope(p.scope_name or "model",
                           reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx()),
                           custom_getter=dtype.float32_variable_storage_getter):
      state = encoder(features["source"], features['to_lang'], p, gater)
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
      step_loss, step_logits, step_state, _, _ = decoder(
        target, state, p, gater)
      del state['time']

      return step_logits, step_state

  beam_outputs = search.beam_search(features, encoding_fn, decoding_fn, p)

  return beam_outputs


# register the model, with a unique name
model.model_register("transformer_multilingual_clsr", train_fn, score_fn, infer_fn)
