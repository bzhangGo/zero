# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import func
from utils import util, dtype


def dot_attention(query, memory, mem_mask, hidden_size,
                  ln=False, num_heads=1, cache=None, dropout=None,
                  out_map=True, scope=None):
    """
    dotted attention model
    :param query: [batch_size, qey_len, dim]
    :param memory: [batch_size, seq_len, mem_dim] or None
    :param mem_mask: [batch_size, seq_len]
    :param hidden_size: attention space dimension
    :param ln: whether use layer normalization
    :param num_heads: attention head number
    :param dropout: attention dropout, default disable
    :param out_map: output additional mapping
    :param cache: cache-based decoding
    :param scope:
    :return: a value matrix, [batch_size, qey_len, mem_dim]
    """
    with tf.variable_scope(scope or "dot_attention", reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx())):
        if memory is None:
            # suppose self-attention from queries alone
            h = func.linear(query, hidden_size * 3, ln=ln, scope="qkv_map")
            q, k, v = tf.split(h, 3, -1)

            if cache is not None:
                k = tf.concat([cache['k'], k], axis=1)
                v = tf.concat([cache['v'], v], axis=1)
                cache = {
                    'k': k,
                    'v': v,
                }
        else:
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

        # q * k => attention weights
        logits = tf.matmul(q, k, transpose_b=True)

        # convert the mask to 0-1 form and multiply to logits
        if mem_mask is not None:
            zero_one_mask = tf.to_float(tf.equal(mem_mask, 0.0))
            logits *= zero_one_mask

        # replace softmax with relu
        # weights = tf.nn.softmax(logits)
        weights = tf.nn.relu(logits)

        dweights = util.valid_apply_dropout(weights, dropout)

        # weights * v => attention vectors
        o = tf.matmul(dweights, v)
        o = func.combine_heads(o)

        # perform RMSNorm to stabilize running
        o = gated_rms_norm(o, scope="post")

        if out_map:
            o = func.linear(o, hidden_size, ln=ln, scope="o_map")

        results = {
            'weights': weights,
            'output': o,
            'cache': cache
        }

        return results


def gated_rms_norm(x, eps=None, scope=None):
    """RMS-based Layer normalization layer"""
    if eps is None:
        eps = dtype.epsilon()
    with tf.variable_scope(scope or "rms_norm",
                           dtype=tf.as_dtype(dtype.floatx())):
        layer_size = util.shape_list(x)[-1]

        scale = tf.get_variable("scale", [layer_size], initializer=tf.ones_initializer())
        gate = tf.get_variable("gate", [layer_size], initializer=None)

        ms = tf.reduce_mean(x ** 2, -1, keep_dims=True)

        # adding gating here which slightly improves quality
        return scale * x * tf.rsqrt(ms + eps) * tf.nn.sigmoid(gate * x)
