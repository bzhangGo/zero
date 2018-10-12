# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from func import additive_attention, linear
from utils import util


def gru(h_, x_g, x_h, d, ln=False):
    # h_: the previous hidden state
    # x_g : the current input state for gate
    # x_h : the current input state for hidden
    # d : the output or internal hidden size
    # ln: whether use layer normalization
    """
        z = sigmoid(h_, x)
        r = sigmoid(h_, x)
        h' = tanh(x, r * h_)
        h = z * h_ + (1. - z) * h'
    """
    h_g = linear(h_, d * 2,
                 ln=ln, scope="gate_h")
    z, r = tf.split(
        tf.sigmoid(x_g + h_g), 2, -1)

    h_h = linear(h_ * r, d,
                 ln=ln, scope="hide_h")
    h = tf.tanh(x_h + h_h)

    h = z * h_ + (1. - z) * h
    return h


def rnn(inputs, mask, hidden_size, init_state=None, ln=False, sm=True):
    in_shape = util.shape_list(inputs)
    batch_size, time_steps = in_shape[:2]

    if init_state is None:
        init_state = tf.zeros([batch_size, hidden_size], tf.float32)

    # prepare projected input
    gate_inputs = linear(inputs, hidden_size * 2,
                         bias=False, ln=ln, scope="gate_x")
    hide_inputs = linear(inputs, hidden_size,
                         bias=False, ln=ln, scope="hide_x")

    gate_ta = tf.transpose(gate_inputs, [1, 0, 2])
    hide_ta = tf.transpose(hide_inputs, [1, 0, 2])
    mask_ta = tf.transpose(tf.expand_dims(mask, -1), [1, 0, 2])

    def _step_fn(prev, x):
        t, h_ = prev
        x_g, x_h, m = x

        h = gru(h_, x_g, x_h, hidden_size, ln=ln)
        h = m * h + (1. - m) * h_

        return t + 1, h

    time = tf.constant(0, dtype=tf.int32, name="time")
    step_states = (time, init_state)
    step_vars = (gate_ta, hide_ta, mask_ta)

    outputs = tf.scan(_step_fn,
                      step_vars,
                      initializer=step_states,
                      parallel_iterations=32,
                      swap_memory=sm)

    output_ta = outputs[1]
    output_state = outputs[1][-1]

    outputs = tf.transpose(output_ta, [1, 0, 2])

    return outputs, output_state


def encoder(source, params):
    mask = tf.to_float(tf.cast(source, tf.bool))
    hidden_size = params.hidden_size

    source, mask = util.remove_invalid_seq(source, mask)

    embed_name = "embedding" if params.shared_source_target_embedding \
        else "src_embedding"
    src_emb = tf.get_variable(embed_name,
                              [params.src_vocab.size(), params.embed_size])
    src_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(src_emb, source)
    inputs = tf.nn.bias_add(inputs, src_bias)

    if util.valid_dropout(params.dropout):
        inputs = tf.nn.dropout(inputs, 1. - params.dropout)

    with tf.variable_scope("encoder"):
        # forward rnn
        with tf.variable_scope('forward'):
            output_fw, state_fw = rnn(inputs, mask,
                                      hidden_size, ln=params.layer_norm,
                                      sm=params.swap_memory)
        # backward rnn
        with tf.variable_scope('backward'):
            output_bw, state_bw = rnn(tf.reverse(inputs, [1]),
                                      tf.reverse(mask, [1]),
                                      hidden_size, ln=params.layer_norm,
                                      sm=params.swap_memory)
            output_bw = tf.reverse(output_bw, [1])

    source_encodes = tf.concat([output_fw, output_bw], -1)
    source_feature = tf.concat([state_fw, state_bw], -1)
    decoder_init = linear(source_feature, hidden_size, ln=params.layer_norm,
                          scope="decoder_initializer")
    decoder_init = tf.tanh(decoder_init)

    return {
        "encodes": source_encodes,
        "decoder_initializer": decoder_init,
        "mask": mask
    }


def cond_rnn(inputs, encodes, hidden_size, init_state=None,
             mask=None, enc_mask=None, ln=False, sm=True):
    in_shape = util.shape_list(inputs)
    batch_size, time_steps = in_shape[:2]
    en_shape = util.shape_list(encodes)

    if init_state is None:
        init_state = tf.zeros([batch_size, hidden_size], tf.float32)
    if mask is None:
        mask = tf.ones([batch_size, time_steps], tf.float32)
    if enc_mask is None:
        enc_mask = tf.ones([batch_size, en_shape[1]], tf.float32)

    # prepare projected encodes and inputs
    gate_inputs = linear(inputs, hidden_size * 2,
                         bias=False, ln=ln, scope="gate_x")
    hide_inputs = linear(inputs, hidden_size,
                         bias=False, ln=ln, scope="hide_x")
    proj_encodes = linear(encodes, hidden_size * 2, bias=False,
                          ln=ln, scope="context_att")

    gate_ta = tf.transpose(gate_inputs, [1, 0, 2])
    hide_ta = tf.transpose(hide_inputs, [1, 0, 2])
    mask_ta = tf.transpose(tf.expand_dims(mask, -1), [1, 0, 2])

    def _step_fn(prev, x):
        t, h_, c_, a_ = prev
        x_g, x_h, m = x

        with tf.variable_scope('lgru'):
            s = gru(h_, x_g, x_h, hidden_size, ln=ln)
            s = m * s + (1. - m) * h_

        a, c = additive_attention(
            s, encodes, enc_mask, hidden_size * 2, ln=ln,
            proj_memory=proj_encodes, scope="attention")

        with tf.variable_scope('hgru'):
            c_g = linear(c, hidden_size * 2,
                         bias=False, ln=ln, scope="gate_c")
            c_h = linear(c, hidden_size,
                         bias=False, ln=ln, scope="hide_c")
            h = gru(s, c_g, c_h, hidden_size, ln=ln)
            h = m * h + (1. - m) * s

        return t + 1, h, c, a

    time = tf.constant(0, dtype=tf.int32, name="time")
    init_context = tf.zeros([batch_size, en_shape[-1]], tf.float32)
    init_weight = tf.zeros([batch_size, en_shape[1]], tf.float32)
    step_states = (time, init_state, init_context, init_weight)
    step_vars = (gate_ta, hide_ta, mask_ta)

    outputs = tf.scan(_step_fn,
                      step_vars,
                      initializer=step_states,
                      parallel_iterations=32,
                      swap_memory=sm)

    output_ta = outputs[1]
    context_ta = outputs[2]
    attention_ta = outputs[3]

    outputs = tf.transpose(output_ta, [1, 0, 2])
    contexts = tf.transpose(context_ta, [1, 0, 2])
    attentions = tf.transpose(attention_ta, [1, 0, 2])

    return outputs, contexts, attentions


def decoder(target, state, params):
    mask = tf.to_float(tf.cast(target, tf.bool))
    hidden_size = params.hidden_size

    if 'decoder' not in state:
        target, mask = util.remove_invalid_seq(target, mask)

    embed_name = "embedding" if params.shared_source_target_embedding \
        else "tgt_embedding"
    tgt_emb = tf.get_variable(embed_name,
                              [params.tgt_vocab.size(), params.embed_size])
    tgt_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(tgt_emb, target)
    inputs = tf.nn.bias_add(inputs, tgt_bias)

    # shift
    if 'decoder' not in state:
        inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])
        inputs = inputs[:, :-1, :]
    else:
        inputs = tf.cond(tf.reduce_all(tf.equal(target, params.tgt_vocab.pad())),
                         lambda: tf.zeros_like(inputs),
                         lambda: inputs)
        mask = tf.ones_like(mask)

    if util.valid_dropout(params.dropout):
        inputs = tf.nn.dropout(inputs, 1. - params.dropout)

    with tf.variable_scope("decoder"):
        init_state = state["decoder_initializer"]
        if 'decoder' in state:
            init_state = state["decoder"]["state"]
        returns = cond_rnn(inputs, state["encodes"], hidden_size,
                           init_state=init_state, mask=mask,
                           enc_mask=state["mask"], ln=params.layer_norm,
                           sm=params.swap_memory)
        outputs, contexts, attentions = returns

    feature = linear([outputs, contexts, inputs], params.embed_size,
                     ln=params.layer_norm, scope="pre_logits")
    feature = tf.tanh(feature)
    if util.valid_dropout(params.dropout):
        feature = tf.nn.dropout(feature, 1. - params.dropout)

    embed_name = "tgt_embedding" if params.shared_target_softmax_embedding \
        else "softmax_embedding"
    embed_name = "embedding" if params.shared_source_target_embedding \
        else embed_name
    softmax_emb = tf.get_variable(embed_name,
                                  [params.tgt_vocab.size(), params.embed_size])
    feature = tf.reshape(feature, [-1, params.embed_size])
    logits = tf.matmul(feature, softmax_emb, False, True)

    centropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=util.label_smooth(target,
                                 util.shape_list(logits)[-1],
                                 factor=params.label_smooth)
    )
    centropy = tf.reshape(centropy, tf.shape(target))

    loss = tf.reduce_sum(centropy * mask, -1) / tf.reduce_sum(mask, -1)
    loss = tf.reduce_mean(loss)

    # these mask tricks mainly used to deal with zero shapes, such as [0, 1]
    loss = tf.cond(tf.equal(tf.shape(target)[0], 0),
                   lambda: tf.constant(0, dtype=tf.float32),
                   lambda: loss)

    if 'decoder' in state:
        state['decoder']['state'] = outputs

    return loss, logits, state


def train_fn(features, params, initializer=None):
    with tf.variable_scope(params.model_name or "model",
                           initializer=initializer,
                           reuse=tf.AUTO_REUSE):
        state = encoder(features['source'], params)
        loss, logits, state = decoder(features['target'], state, params)
        return loss


def infer_fn(params):
    params = copy.copy(params)
    params = util.closing_dropout(params)

    def encoding_fn(source):
        with tf.variable_scope(params.model_name or "model",
                               reuse=tf.AUTO_REUSE):
            state = encoder(source, params)
            state["decoder"] = {
                "state": state["decoder_initializer"]
            }
            return state

    def decoding_fn(target, state):
        with tf.variable_scope(params.model_name or "model",
                               reuse=tf.AUTO_REUSE):
            step_loss, step_logits, step_state = decoder(
                target, state, params)
            step_state["decoder"]["state"] = util.merge_neighbor_dims(
                step_state["decoder"]["state"], axis=0
            )
            return step_logits, step_state

    return encoding_fn, decoding_fn
