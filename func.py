# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import util


def linear(x, dim, bias=True, ln=False,
           weight_initializer=None,
           bias_initializer=None,
           scope=None):
    """
    basic linear or feed forward layer
    :param x: input tensor or list
    :param dim: output dimension or list
    :param bias: whether use bias term
    :param ln: whether use layer normalization
    :param weight_initializer: you can set it if you want
    :param bias_initializer: you can set it if you want
    :param scope
    :return:
    """
    with tf.variable_scope(scope or "linear", values=[x]):
        if not isinstance(x, (list, tuple)):
            x = [x]
        if not isinstance(dim, (list, tuple)):
            dim = [dim]

        if not ln:
            # by default, we concatenate inputs
            x = [tf.concat(x, -1)]

        outputs = []
        for oidx, osize in enumerate(dim):

            results = []
            for iidx, ix in enumerate(x):
                x_shp = util.shape_list(ix)
                xsize = x_shp[-1]

                W = tf.get_variable(
                    "W_{}_{}".format(oidx, iidx), [xsize, osize],
                    initializer=weight_initializer)
                o = tf.matmul(tf.reshape(ix, [-1, xsize]), W)

                if ln:
                    o = layer_norm(
                        o, scope="ln_{}_{}".format(oidx, iidx))
                results.append(o)

            o = tf.add_n(results)

            if bias:
                b = tf.get_variable(
                    "b_{}".format(oidx), [osize],
                    initializer=bias_initializer)
                o = tf.nn.bias_add(o, b)
            x_shp = util.shape_list(x[0])[:-1]
            o = tf.reshape(o, tf.concat([x_shp, [osize]], 0))

            outputs.append(o)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


def additive_attention(query, memory, mem_mask,
                       hidden_size, ln=False, proj_memory=None,
                       scope=None):
    """
    additive attention model
    :param query: [batch_size, dim]
    :param memory: [batch_size, seq_len, mem_dim]
    :param mem_mask: [batch_size, seq_len]
    :param hidden_size: attention space dimension
    :param ln: whether use layer normalization
    :param proj_memory: this is the mapped memory for saving memory
    :param scope:
    :return: a value matrix, [batch_size, mem_dim]
    """
    with tf.variable_scope(scope or "additive_attention"):
        if proj_memory is None:
            proj_memory = linear(
                memory, hidden_size, ln=ln, scope="feed_memory")

        value = linear(tf.expand_dims(query, 1),
                       hidden_size, ln=ln, scope="feed_query") + \
                proj_memory

        value = tf.tanh(value)

        logits = linear(value, 1, ln=False, scope="feed_logits")
        logits = tf.squeeze(logits, -1)
        logits = util.mask_scale(logits, mem_mask)

        weights = tf.nn.softmax(logits, 1)  # [batch_size, seq_len]
        value = tf.reduce_sum(tf.expand_dims(weights, -1) * memory, 1)

        return weights, value


def layer_norm(x, eps=1e-8, scope=None):
    """Layer normalization layer"""
    with tf.variable_scope(scope or "layer_norm"):
        layer_size = util.shape_list(x)[-1]

        scale = tf.get_variable("scale", [layer_size],
                                initializer=tf.ones_initializer())
        offset = tf.get_variable("offset", [layer_size],
                                 initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(x, -1, keep_dims=True)
        var = tf.reduce_mean((x - mean) ** 2, -1, keep_dims=True)

        return scale * (x - mean) * tf.rsqrt(var + eps) + offset
