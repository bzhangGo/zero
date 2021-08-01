# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import func
from utils import util, dtype

import tensorflow as tf

"""Conditional Computation Module

CCT: Controlling Computation versus Quality for Neural Sequence Models
Link: https://arxiv.org/pdf/2002.07106.pdf
Code reference: https://github.com/tensorflow/lingvo

# 1. linear schedule for the cct noise \alpha
# 2. gating layer with noise for training or evaluation
# 3. cct: feedfoward layer
# 4. cct: attention layer

See `zero/models/transformer_cct.py` to see how we implement the CCT model.
"""


class CCTGater(object):
  """Conditional Computational Gating Layer

  Provide binary gates with noises during training. Generally, the gater works as follows:
    - during training, y = `cctgater(x)` * f(x) + x
    - during inference, y = f(x) + x if `cctgater(x) > 0.5` else y = x

  """
  def __init__(self, alpha_value, total_step, is_training, middle_dim):
    self._alpha_value = alpha_value
    self._total_step = total_step
    self._is_training = is_training
    self._middle_dim = middle_dim

  def get_alpha(self):
    # linear schedule: steps 0 -> _total_step; alpha 0 -> _alpha_value
    global_step = tf.train.get_or_create_global_step()
    _alpha = tf.cast(global_step, tf.float32) * self._alpha_value / self._total_step
    return tf.minimum(_alpha, self._alpha_value)

  def gating_layer(self, x, n_out, scope="gating"):
    # x: input, [batch, len, dim]
    # n_out: output dimension
    # alpha: noise factor

    h = func.linear(x, self._middle_dim, scope="%s_middle" % scope)
    p_c = func.linear(tf.nn.relu(h), n_out, scope=scope)
  
    if self._is_training:
      noise = tf.random.normal(util.shape_list(p_c))
      p_c = tf.nn.sigmoid(p_c + self.get_alpha() * noise)
    else:
      ones = tf.ones_like(p_c)
      zeros = tf.zeros_like(p_c)
      p_c = tf.where(
        tf.greater_equal(p_c, tf.constant(0.0, dtype=p_c.dtype)),
        ones, zeros)
    return p_c


def cct_budget(gates, query_mask, memory_mask):
  """Compute how many tokens are retrained by the gates

  By collecting sum(`gate` * mask)/sum(mask), we get statistics about how many
  computation we used in our model. We treat sum(`gate` * mask) as the actual used
  budget, and sum(mask) as the total budget. In CCT, we use a `cct_bucket_p` to
  regularize the budget we used in Transformer.

  mask => the mask upon each batch/token, indicating valid tokens
  query_mask: for FFN layer, self-attention, query part of encdec-attention.
  memory_mask: for memory part of encdec attention alone
  gates: dictionary, contains every kinds of `p_c` matrix
  """

  bgsum, bgall = 0., 0.
  query_mask = tf.expand_dims(query_mask, -1)
  memory_mask = tf.expand_dims(memory_mask, -1)

  for key in gates:
    value = gates[key]

    if value is None:
      continue

    if "san" in key:
      # self attention: batch x sequence x 1
      mask = query_mask
    elif "can" in key:
      # cross attention: batch x sequence x 1
      if "query" in key:
        mask = query_mask
      else:
        assert "memory" in key, 'Memory should exist!'
        mask = memory_mask
    else:
      assert "ffn" in key, "FFn should exist!"
      mask = query_mask

    bgsum += tf.reduce_sum(value * mask)
    bgall += tf.reduce_sum(tf.ones_like(value) * mask)

  return bgsum, bgall


def cct_ffn_layer(x, d, d_o, M, gater, dropout=None, scope=None):
  """Conditional FFN layer in Transformer

  Split middle layer into `M` parts, and for each part, CCT assigns a binary gate to
  indicate whether or not use it.
  """
  with tf.variable_scope(scope or "cct_ffn_layer",
                         dtype=tf.as_dtype(dtype.floatx())):
    outputs = []

    for i in range(M):
      ffn_out = func.ffn_layer(x, d // M, d_o, dropout=dropout, scope="cct_ffn_sublayer_%d" % i)
      normed_ffn_out = func.layer_norm(ffn_out, scope="cct_ffn_norm_%d" % i)

      outputs.append(normed_ffn_out)

    # batch x len x M x d_o
    output = tf.stack(outputs, axis=2)
    # batch x len x M
    ffn_p_c = gater.gating_layer(x, M, scope="ffn_gating")

    output = output * tf.expand_dims(ffn_p_c, -1)
    output = tf.reduce_sum(output, axis=2)

    return output, ffn_p_c


def cct_dot_attention(query, memory, mem_mask, hidden_size, gater,
                      ln=False, num_heads=1, cache=None, dropout=None,
                      out_map=True, scope=None, fuse_mask=None,
                      decode_step=None):
  """Conditional Multi-Head Attention in Transformer

  CCT assigns binary gates to query and memory, respectively.
    - Gate memory before computing key/value;
    - Gate query after attention

  Difference compared to the original implementation: we didn't apply gating layer to
  the memory (the same to query) of self-attention layers.
  """

  with tf.variable_scope(scope or "cct_dot_attention", reuse=tf.AUTO_REUSE,
                         dtype=tf.as_dtype(dtype.floatx())):
    if fuse_mask is not None:
      assert memory is not None, 'Fuse mechanism only applied with cross-attention'

    # apply conditional gating
    memory_p_c = None
    query_p_c = gater.gating_layer(query, 1, scope="query_gating")

    if memory is None:
      # suppose self-attention from queries alone
      h = func.linear(query, hidden_size * 3, ln=ln, scope="qkv_map")
      q, k, v = tf.split(h, 3, -1)

      if cache is not None:
        k = util.state_time_insert(cache['k'], k, decode_step, axis=1)
        v = util.state_time_insert(cache['v'], v, decode_step, axis=1)
        cache = {
          'k': k,
          'v': v,
        }
    else:
      # source-target cross attention
      memory_p_c = gater.gating_layer(memory, 1, scope="memory_gating")
      memory *= memory_p_c

      q = func.linear(query, hidden_size, ln=ln, scope="q_map")
      if cache is not None and ('mk' in cache and 'mv' in cache):
        k, v = cache['mk'], cache['mv']
      else:
        k = func.linear(memory, hidden_size, ln=ln, scope="k_map")
        v = func.linear(memory, hidden_size, ln=ln, scope="v_map")

      if cache is not None:
        cache['mk'] = k
        cache['mv'] = v

    q = func.split_heads(q, num_heads)
    k = func.split_heads(k, num_heads)
    v = func.split_heads(v, num_heads)

    q *= (hidden_size // num_heads) ** (-0.5)

    logits = tf.matmul(q, k, transpose_b=True)
    if mem_mask is not None:
      logits += mem_mask

    weights = tf.nn.softmax(logits)
    dweights = util.valid_apply_dropout(weights, dropout)

    # weights * v => attention vectors
    o = tf.matmul(dweights, v)

    o = func.combine_heads(o)

    if fuse_mask is not None:
      # This is for AAN, the important part is sharing v_map
      v_q = func.linear(query, hidden_size, ln=ln, scope="v_map")

      if cache is not None and 'aan' in cache:
        aan_o = (v_q + cache['aan']) / dtype.tf_to_float(fuse_mask + 1)
      else:
        # Simplified Average Attention Network
        aan_o = tf.matmul(fuse_mask, v_q)

      if cache is not None:
        if 'aan' not in cache:
          cache['aan'] = v_q
        else:
          cache['aan'] = v_q + cache['aan']
        cache['aan'] = tf.reshape(cache['aan'], [util.shape_list(query)[0], 1, hidden_size])

      # Directly sum both self-attention and cross attention
      o = o + aan_o

    if out_map:
      o = func.linear(o, hidden_size, ln=ln, scope="o_map")

    # adding gating layer
    o = query_p_c * func.layer_norm(o, scope="cct_post_norm")

    results = {
      'weights': weights,
      'output': o,
      'cache': cache,
      'query_p_c': query_p_c,
      'memory_p_c': memory_p_c,
    }

    return results
