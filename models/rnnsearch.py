# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from func import linear
from utils import util
from rnns import rnn


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
            outputs = rnn.rnn(params.cell, inputs, hidden_size, mask=mask,
                              ln=params.layer_norm, sm=params.swap_memory)
            output_fw, state_fw = outputs[1]
        # backward rnn
        with tf.variable_scope('backward'):
            if not params.caencoder:
                outputs = rnn.rnn(params.cell, tf.reverse(inputs, [1]),
                                  hidden_size, mask=tf.reverse(mask, [1]),
                                  ln=params.layer_norm, sm=params.swap_memory)
                output_bw, state_bw = outputs[1]
            else:
                outputs = rnn.cond_rnn(params.cell, tf.reverse(inputs, [1]),
                                       tf.reverse(output_fw, [1]), hidden_size,
                                       mask=tf.reverse(mask, [1]),
                                       ln=params.layer_norm,
                                       sm=params.swap_memory,
                                       one2one=True)
                output_bw, state_bw = outputs[1]

            output_bw = tf.reverse(output_bw, [1])

    if not params.caencoder:
        source_encodes = tf.concat([output_fw, output_bw], -1)
        source_feature = tf.concat([state_fw, state_bw], -1)
    else:
        source_encodes = output_bw
        source_feature = state_bw

    with tf.variable_scope("decoder_initializer"):
        decoder_init = rnn.get_cell(
            params.cell, hidden_size, ln=params.layer_norm
        ).get_init_state(x=source_feature)
    decoder_init = tf.tanh(decoder_init)

    return {
        "encodes": source_encodes,
        "decoder_initializer": decoder_init,
        "mask": mask
    }


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
        returns = rnn.cond_rnn(params.cell, inputs, state["encodes"], hidden_size,
                               init_state=init_state, mask=mask,
                               mem_mask=state["mask"], ln=params.layer_norm,
                               sm=params.swap_memory, one2one=False)
        (hidden_states, _), (outputs, _), contexts, attentions = returns

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
        state['decoder']['state'] = hidden_states

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

    def decoding_fn(target, state, time):
        with tf.variable_scope(params.model_name or "model",
                               reuse=tf.AUTO_REUSE):
            step_loss, step_logits, step_state = decoder(
                target, state, params)
            step_state["decoder"]["state"] = util.merge_neighbor_dims(
                step_state["decoder"]["state"], axis=0
            )
            return step_logits, step_state

    return encoding_fn, decoding_fn
