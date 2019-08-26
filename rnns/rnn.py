# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import util, dtype
from rnns import get_cell
from func import linear, additive_attention


def rnn(cell_name, x, d, mask=None, ln=False, init_state=None, sm=True):
    """Self implemented RNN procedure, supporting mask trick"""
    # cell_name: gru, lstm or atr
    # x: input sequence embedding matrix, [batch, seq_len, dim]
    # d: hidden dimension for rnn
    # mask: mask matrix, [batch, seq_len]
    # ln: whether use layer normalization
    # init_state: the initial hidden states, for cache purpose
    # sm: whether apply swap memory during rnn scan
    # dp: variational dropout

    in_shape = util.shape_list(x)
    batch_size, time_steps = in_shape[:2]

    cell = get_cell(cell_name, d, ln=ln)

    if init_state is None:
        init_state = cell.get_init_state(shape=[batch_size])
    if mask is None:
        mask = dtype.tf_to_float(tf.ones([batch_size, time_steps]))

    # prepare projected input
    cache_inputs = cell.fetch_states(x)
    cache_inputs = [tf.transpose(v, [1, 0, 2])
                    for v in list(cache_inputs)]
    mask_ta = tf.transpose(tf.expand_dims(mask, -1), [1, 0, 2])

    def _step_fn(prev, x):
        t, h_ = prev
        m = x[-1]
        v = x[:-1]

        h = cell(h_, v)
        h = m * h + (1. - m) * h_

        return t + 1, h

    time = tf.constant(0, dtype=tf.int32, name="time")
    step_states = (time, init_state)
    step_vars = cache_inputs + [mask_ta]

    outputs = tf.scan(_step_fn,
                      step_vars,
                      initializer=step_states,
                      parallel_iterations=32,
                      swap_memory=sm)

    output_ta = outputs[1]
    output_state = outputs[1][-1]

    outputs = tf.transpose(output_ta, [1, 0, 2])

    return (outputs, output_state), \
           (cell.get_hidden(outputs), cell.get_hidden(output_state))


def cond_rnn(cell_name, x, memory, d, init_state=None,
             mask=None, mem_mask=None, ln=False, sm=True,
             one2one=False, num_heads=1):
    """Self implemented conditional-RNN procedure, supporting mask trick"""
    # cell_name: gru, lstm or atr
    # x: input sequence embedding matrix, [batch, seq_len, dim]
    # memory: the conditional part
    # d: hidden dimension for rnn
    # mask: mask matrix, [batch, seq_len]
    # mem_mask: memory mask matrix, [batch, mem_seq_len]
    # ln: whether use layer normalization
    # init_state: the initial hidden states, for cache purpose
    # sm: whether apply swap memory during rnn scan
    # one2one: whether the memory is one-to-one mapping for x
    # num_heads: number of attention heads, multi-head attention
    # dp: variational dropout

    in_shape = util.shape_list(x)
    batch_size, time_steps = in_shape[:2]
    mem_shape = util.shape_list(memory)

    cell_lower = get_cell(cell_name, d, ln=ln,
                          scope="{}_lower".format(cell_name))
    cell_higher = get_cell(cell_name, d, ln=ln,
                           scope="{}_higher".format(cell_name))

    if init_state is None:
        init_state = cell_lower.get_init_state(shape=[batch_size])
    if mask is None:
        mask = dtype.tf_to_float(tf.ones([batch_size, time_steps]))
    if mem_mask is None:
        mem_mask = dtype.tf_to_float(tf.ones([batch_size, mem_shape[1]]))

    # prepare projected encodes and inputs
    cache_inputs = cell_lower.fetch_states(x)
    cache_inputs = [tf.transpose(v, [1, 0, 2])
                    for v in list(cache_inputs)]
    if not one2one:
        proj_memories = linear(memory, mem_shape[-1], bias=False,
                               ln=ln, scope="context_att")
    else:
        cache_memories = cell_higher.fetch_states(memory)
        cache_memories = [tf.transpose(v, [1, 0, 2])
                          for v in list(cache_memories)]
    mask_ta = tf.transpose(tf.expand_dims(mask, -1), [1, 0, 2])
    init_context = dtype.tf_to_float(tf.zeros([batch_size, mem_shape[-1]]))
    init_weight = dtype.tf_to_float(tf.zeros([batch_size, num_heads, mem_shape[1]]))
    mask_pos = len(cache_inputs)

    def _step_fn(prev, x):
        t, h_, c_, a_ = prev

        if not one2one:
            m, v = x[mask_pos], x[:mask_pos]
        else:
            c, c_c, m, v = x[-1], x[mask_pos+1:-1], x[mask_pos], x[:mask_pos]

        s = cell_lower(h_, v)
        s = m * s + (1. - m) * h_

        if not one2one:
            vle = additive_attention(
                cell_lower.get_hidden(s), memory, mem_mask,
                mem_shape[-1], ln=ln, num_heads=num_heads,
                proj_memory=proj_memories, scope="attention")
            a, c = vle['weights'], vle['output']
            c_c = cell_higher.fetch_states(c)
        else:
            a = tf.tile(tf.expand_dims(tf.range(time_steps), 0), [batch_size, 1])
            a = dtype.tf_to_float(tf.equal(a, t))
            a = tf.tile(tf.expand_dims(a, 1), [1, num_heads, 1])
            a = tf.reshape(a, tf.shape(init_weight))

        h = cell_higher(s, c_c)
        h = m * h + (1. - m) * s

        return t + 1, h, c, a

    time = tf.constant(0, dtype=tf.int32, name="time")
    step_states = (time, init_state, init_context, init_weight)
    step_vars = cache_inputs + [mask_ta]
    if one2one:
        step_vars += cache_memories + [memory]

    outputs = tf.scan(_step_fn,
                      step_vars,
                      initializer=step_states,
                      parallel_iterations=32,
                      swap_memory=sm)

    output_ta = outputs[1]
    context_ta = outputs[2]
    attention_ta = outputs[3]

    outputs = tf.transpose(output_ta, [1, 0, 2])
    output_states = outputs[:, -1]
    contexts = tf.transpose(context_ta, [1, 0, 2])
    attentions = tf.transpose(attention_ta, [1, 2, 0, 3])

    return (outputs, output_states), \
           (cell_higher.get_hidden(outputs), cell_higher.get_hidden(output_states)), \
        contexts, attentions
