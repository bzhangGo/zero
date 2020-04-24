# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

import func
from models import model
from utils import util, dtype


def lang_mapper(x, to_lang, lang_size):
    """Mapping x with language-specific modeling"""
    x_size = util.shape_list(x)[-1]
    # we tried to relax this mapping by factoring this matrix, but not work
    W_lang = tf.get_variable("lang_mapper", [lang_size, x_size * x_size])

    W = tf.reshape(tf.gather(W_lang, to_lang), [-1, x_size, x_size])
    o = tf.einsum('bsi,bij->bsj', x, W)

    return o


def encoder(source, to_lang, params):
    mask = dtype.tf_to_float(tf.cast(source, tf.bool))
    hidden_size = params.hidden_size
    initializer = tf.random_normal_initializer(0.0, hidden_size ** -0.5)

    source, mask = util.remove_invalid_seq(source, mask)

    embed_name = "embedding" if params.shared_source_target_embedding \
        else "src_embedding"
    src_emb = tf.get_variable(embed_name,
                              [params.src_vocab.size(), params.embed_size],
                              initializer=initializer)
    src_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(src_emb, source) * (hidden_size ** 0.5)
    inputs = tf.nn.bias_add(inputs, src_bias)
    inputs = func.add_timing_signal(inputs)

    inputs = util.valid_apply_dropout(inputs, params.dropout)

    with tf.variable_scope("encoder"):
        x = inputs
        for layer in range(params.num_encoder_layer):
            if params.deep_transformer_init:
                layer_initializer = tf.variance_scaling_initializer(
                    params.initializer_gain * (layer + 1) ** -0.5,
                    mode="fan_avg",
                    distribution="uniform")
            else:
                layer_initializer = None
            with tf.variable_scope("layer_{}".format(layer), initializer=layer_initializer):
                with tf.variable_scope("self_attention"):
                    y = func.dot_attention(
                        x,
                        None,
                        func.attention_bias(mask, "masking"),
                        hidden_size,
                        num_heads=params.num_heads,
                        dropout=params.attention_dropout
                    )

                    y = y['output']
                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    # language-aware layer normalization
                    x = func.layer_norm(x, lang=to_lang, lang_size=params.to_lang_vocab.size())

                with tf.variable_scope("feed_forward"):
                    y = func.ffn_layer(
                        x,
                        params.filter_size,
                        hidden_size,
                        dropout=params.relu_dropout,
                    )

                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    # language-aware layer normalization
                    x = func.layer_norm(x, lang=to_lang, lang_size=params.to_lang_vocab.size())

        # language-aware layer mapping
        x = lang_mapper(x, to_lang, params.to_lang_vocab.size())

    source_encodes = x
    x_shp = util.shape_list(x)

    return {
        "encodes": source_encodes,
        "decoder_initializer": {
            "layer_{}".format(l): {
                # for fusion
                "aan": dtype.tf_to_float(tf.zeros([x_shp[0], 1, hidden_size])),
                "k": dtype.tf_to_float(tf.zeros([x_shp[0], 0, hidden_size])),
                # for transformer
                "v": dtype.tf_to_float(tf.zeros([x_shp[0], 0, hidden_size])),
            }
            for l in range(params.num_decoder_layer)
        },
        "mask": mask,
        "to_lang": to_lang,
    }


def decoder(target, to_lang, state, params):
    mask = dtype.tf_to_float(tf.cast(target, tf.bool))
    hidden_size = params.hidden_size
    initializer = tf.random_normal_initializer(0.0, hidden_size ** -0.5)

    is_training = ('decoder' not in state)

    if is_training:
        target, mask = util.remove_invalid_seq(target, mask)

    embed_name = "embedding" if params.shared_source_target_embedding \
        else "tgt_embedding"
    tgt_emb = tf.get_variable(embed_name,
                              [params.tgt_vocab.size(), params.embed_size],
                              initializer=initializer)
    tgt_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(tgt_emb, target) * (hidden_size ** 0.5)
    inputs = tf.nn.bias_add(inputs, tgt_bias)

    # shift
    if is_training:
        inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])
        inputs = inputs[:, :-1, :]
        inputs = func.add_timing_signal(inputs)
    else:
        inputs = tf.cond(tf.reduce_all(tf.equal(target, params.tgt_vocab.pad())),
                         lambda: tf.zeros_like(inputs),
                         lambda: inputs)
        mask = tf.ones_like(mask)
        inputs = func.add_timing_signal(inputs, time=dtype.tf_to_float(state['time']))

    inputs = util.valid_apply_dropout(inputs, params.dropout)

    with tf.variable_scope("decoder"):
        x = inputs
        for layer in range(params.num_decoder_layer):
            if params.deep_transformer_init:
                layer_initializer = tf.variance_scaling_initializer(
                    params.initializer_gain * (layer + 1) ** -0.5,
                    mode="fan_avg",
                    distribution="uniform")
            else:
                layer_initializer = None
            with tf.variable_scope("layer_{}".format(layer), initializer=layer_initializer):
                # by default, we employ merged attention, for faster training and generation
                if params.enable_fuse:
                    with tf.variable_scope("fuse_attention"):
                        y = func.dot_attention(
                            x,
                            state['encodes'],
                            func.attention_bias(state['mask'], "masking"),
                            hidden_size,
                            num_heads=params.num_heads,
                            dropout=params.attention_dropout,
                            fuse_mask=func.attention_bias(mask, "aan") if is_training else state['time'],
                            cache=None if is_training else
                            state['decoder']['state']['layer_{}'.format(layer)]
                        )
                        if not is_training:
                            # mk, mv, aan
                            state['decoder']['state']['layer_{}'.format(layer)]\
                                .update(y['cache'])

                        y = y['output']
                        x = func.residual_fn(x, y, dropout=params.residual_dropout)
                        # language-aware layer normalization
                        x = func.layer_norm(x, lang=to_lang, lang_size=params.to_lang_vocab.size())
                else:
                    with tf.variable_scope("self_attention"):
                        y = func.dot_attention(
                            x,
                            None,
                            func.attention_bias(tf.shape(mask)[1], "causal"),
                            hidden_size,
                            num_heads=params.num_heads,
                            dropout=params.attention_dropout,
                            cache=None if is_training else
                            state['decoder']['state']['layer_{}'.format(layer)]
                        )
                        if not is_training:
                            # k, v
                            state['decoder']['state']['layer_{}'.format(layer)] \
                                .update(y['cache'])

                        y = y['output']
                        x = func.residual_fn(x, y, dropout=params.residual_dropout)
                        # language-aware layer normalization
                        x = func.layer_norm(x, lang=to_lang, lang_size=params.to_lang_vocab.size())

                    with tf.variable_scope("cross_attention"):
                        y = func.dot_attention(
                            x,
                            state['encodes'],
                            func.attention_bias(state['mask'], "masking"),
                            hidden_size,
                            num_heads=params.num_heads,
                            dropout=params.attention_dropout,
                            cache=None if is_training else
                            state['decoder']['state']['layer_{}'.format(layer)]
                        )
                        if not is_training:
                            # mk, mv
                            state['decoder']['state']['layer_{}'.format(layer)] \
                                .update(y['cache'])

                        y = y['output']
                        x = func.residual_fn(x, y, dropout=params.residual_dropout)
                        # language-aware layer normalization
                        x = func.layer_norm(x, lang=to_lang, lang_size=params.to_lang_vocab.size())

                with tf.variable_scope("feed_forward"):
                    y = func.ffn_layer(
                        x,
                        params.filter_size,
                        hidden_size,
                        dropout=params.relu_dropout,
                    )

                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    # language-aware layer normalization
                    x = func.layer_norm(x, lang=to_lang, lang_size=params.to_lang_vocab.size())
    feature = x
    if 'dev_decode' in state:
        feature = x[:, -1, :]

    embed_name = "tgt_embedding" if params.shared_target_softmax_embedding \
        else "softmax_embedding"
    embed_name = "embedding" if params.shared_source_target_embedding \
        else embed_name
    softmax_emb = tf.get_variable(embed_name,
                                  [params.tgt_vocab.size(), params.embed_size],
                                  initializer=initializer)
    feature = tf.reshape(feature, [-1, params.embed_size])
    logits = tf.matmul(feature, softmax_emb, False, True)

    logits = tf.cast(logits, tf.float32)

    soft_label, normalizer = util.label_smooth(
        target,
        util.shape_list(logits)[-1],
        factor=params.label_smooth)
    centropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=soft_label
    )
    centropy -= normalizer
    centropy = tf.reshape(centropy, tf.shape(target))

    mask = tf.cast(mask, tf.float32)
    per_sample_loss = tf.reduce_sum(centropy * mask, -1) / tf.reduce_sum(mask, -1)
    loss = tf.reduce_mean(per_sample_loss)

    # these mask tricks mainly used to deal with zero shapes, such as [0, 1]
    loss = tf.cond(tf.equal(tf.shape(target)[0], 0),
                   lambda: tf.constant(0, dtype=tf.float32),
                   lambda: loss)

    return loss, logits, state, per_sample_loss


def train_fn(features, params, initializer=None):
    with tf.variable_scope(params.scope_name or "model",
                           initializer=initializer,
                           reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx()),
                           custom_getter=dtype.float32_variable_storage_getter):
        state = encoder(features['source'], features['to_lang'], params)
        loss, logits, state, _ = decoder(features['target'], features['to_lang'], state, params)

        return {
            "loss": loss
        }


def score_fn(features, params, initializer=None):
    params = copy.copy(params)
    params = util.closing_dropout(params)
    params.label_smooth = 0.0
    with tf.variable_scope(params.scope_name or "model",
                           initializer=initializer,
                           reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx()),
                           custom_getter=dtype.float32_variable_storage_getter):
        state = encoder(features['source'], features['to_lang'], params)
        _, _, _, scores = decoder(features['target'], features['to_lang'], state, params)

        return {
            "score": scores
        }


def infer_fn(params):
    params = copy.copy(params)
    params = util.closing_dropout(params)

    def encoding_fn(source, to_lang):
        with tf.variable_scope(params.scope_name or "model",
                               reuse=tf.AUTO_REUSE,
                               dtype=tf.as_dtype(dtype.floatx()),
                               custom_getter=dtype.float32_variable_storage_getter):
            state = encoder(source, to_lang, params)
            state["decoder"] = {
                "state": state["decoder_initializer"]
            }
            return state

    def decoding_fn(target, to_lang, state, time):
        with tf.variable_scope(params.scope_name or "model",
                               reuse=tf.AUTO_REUSE,
                               dtype=tf.as_dtype(dtype.floatx()),
                               custom_getter=dtype.float32_variable_storage_getter):
            if params.search_mode == "cache":
                state['time'] = time
                step_loss, step_logits, step_state, _ = decoder(
                    target, to_lang, state, params)
                del state['time']
            else:
                estate = encoder(state, to_lang, params)
                estate['dev_decode'] = True
                _, step_logits, _, _ = decoder(target, to_lang, estate, params)
                step_state = state

            return step_logits, step_state

    return encoding_fn, decoding_fn


# register the model, with a unique name
model.model_register("transformer_multilingual", train_fn, score_fn, infer_fn)
