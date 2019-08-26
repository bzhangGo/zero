# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import func
from utils import util, dtype
from modules import rpr, initializer


def shift_layer(x, scope="shift"):
    with tf.variable_scope(scope or "shift"):
        offset = tf.get_variable("offset", [1], initializer=tf.zeros_initializer())
        return x - offset


def scale_layer(x, init=1., scope="scale"):
    with tf.variable_scope(scope or "scale"):
        scale = tf.get_variable(
            "scale", [1],
            initializer=initializer.scale_initializer(init, tf.ones_initializer()))
        return x * scale


def ffn_layer(x, d, d_o, dropout=None, scope=None, numblocks=None):
    """
    FFN layer in Transformer
    :param numblocks: size of 'L' in fixup paper
    :param scope:
    """
    with tf.variable_scope(scope or "ffn_layer",
                           dtype=tf.as_dtype(dtype.floatx())) as scope:
        assert numblocks is not None, 'Fixup requires the total model depth L'

        in_initializer = initializer.scale_initializer(
            math.pow(numblocks, -1. / 2.), scope.initializer)

        x = shift_layer(x)
        hidden = func.linear(x, d, scope="enlarge",
                             weight_initializer=in_initializer, bias=False)
        hidden = shift_layer(hidden)
        hidden = tf.nn.relu(hidden)

        hidden = util.valid_apply_dropout(hidden, dropout)

        hidden = shift_layer(hidden)
        output = func.linear(hidden, d_o, scope="output", bias=False,
                             weight_initializer=tf.zeros_initializer())
        output = scale_layer(output)

        return output


def dot_attention(query, memory, mem_mask, hidden_size,
                  ln=False, num_heads=1, cache=None, dropout=None,
                  use_relative_pos=False, max_relative_position=16,
                  out_map=True, scope=None, fuse_mask=None,
                  decode_step=None, numblocks=None):
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
    :param fuse_mask: aan mask during training, and timestep for testing
    :param max_relative_position: maximum position considered for relative embedding
    :param use_relative_pos: whether use relative position information
    :param decode_step: the time step of current decoding, 0-based
    :param numblocks: size of 'L' in fixup paper
    :param scope:
    :return: a value matrix, [batch_size, qey_len, mem_dim]
    """
    with tf.variable_scope(scope or "dot_attention", reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx())) as scope:
        if fuse_mask:
            assert memory is not None, 'Fuse mechanism only applied with cross-attention'
        if cache and use_relative_pos:
            assert decode_step is not None, 'Decode Step must provide when use relative position encoding'

        assert numblocks is not None, 'Fixup requires the total model depth L'

        scale_base = 6. if fuse_mask is None else 8.
        in_initializer = initializer.scale_initializer(
            math.pow(numblocks, -1. / scale_base), scope.initializer)

        if memory is None:
            # suppose self-attention from queries alone
            h = func.linear(query, hidden_size * 3, ln=ln, scope="qkv_map",
                            weight_initializer=in_initializer, bias=False)
            q, k, v = tf.split(h, 3, -1)

            if cache is not None:
                k = tf.concat([cache['k'], k], axis=1)
                v = tf.concat([cache['v'], v], axis=1)
                cache = {
                    'k': k,
                    'v': v,
                }
        else:
            q = func.linear(query, hidden_size, ln=ln, scope="q_map",
                            weight_initializer=in_initializer, bias=False)
            if cache is not None and ('mk' in cache and 'mv' in cache):
                k, v = cache['mk'], cache['mv']
            else:
                k = func.linear(memory, hidden_size, ln=ln, scope="k_map",
                                weight_initializer=in_initializer, bias=False)
                v = func.linear(memory, hidden_size, ln=ln, scope="v_map",
                                weight_initializer=in_initializer, bias=False)

            if cache is not None:
                cache['mk'] = k
                cache['mv'] = v

        q = func.split_heads(q, num_heads)
        k = func.split_heads(k, num_heads)
        v = func.split_heads(v, num_heads)

        q *= (hidden_size // num_heads) ** (-0.5)

        q_shp = util.shape_list(q)
        k_shp = util.shape_list(k)
        v_shp = util.shape_list(v)

        q_len = q_shp[2] if decode_step is None else decode_step + 1
        r_lst = None if decode_step is None else 1

        # q * k => attention weights
        if use_relative_pos:
            r = rpr.get_relative_positions_embeddings(
                q_len, k_shp[2], k_shp[3],
                max_relative_position, name="rpr_keys", last=r_lst)
            logits = rpr.relative_attention_inner(q, k, r, transpose=True)
        else:
            logits = tf.matmul(q, k, transpose_b=True)

        if mem_mask is not None:
            logits += mem_mask

        weights = tf.nn.softmax(logits)

        dweights = util.valid_apply_dropout(weights, dropout)

        # weights * v => attention vectors
        if use_relative_pos:
            r = rpr.get_relative_positions_embeddings(
                q_len, k_shp[2], v_shp[3],
                max_relative_position, name="rpr_values", last=r_lst)
            o = rpr.relative_attention_inner(dweights, v, r, transpose=False)
        else:
            o = tf.matmul(dweights, v)

        o = func.combine_heads(o)

        if fuse_mask is not None:
            # This is for AAN, the important part is sharing v_map
            v_q = func.linear(query, hidden_size, ln=ln, scope="v_map",
                              weight_initializer=in_initializer, bias=False)

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

            # Directly sum both self-attention and cross attention
            o = o + aan_o

        if out_map:
            o = func.linear(o, hidden_size, ln=ln, scope="o_map",
                            weight_initializer=tf.zeros_initializer(), bias=False)

        results = {
            'weights': weights,
            'output': o,
            'cache': cache
        }

        return results
