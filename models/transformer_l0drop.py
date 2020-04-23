# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

import func
from models import model
from utils import util, dtype
from modules import l0norm


def dot_attention(query, memory, mem_mask, hidden_size,
                  ln=False, num_heads=1, cache=None, dropout=None,
                  out_map=True, scope=None, count_mask=None):
    """
    dotted attention model with l0drop
    :param query: [batch_size, qey_len, dim]
    :param memory: [batch_size, seq_len, mem_dim] or None
    :param mem_mask: [batch_size, seq_len]
    :param hidden_size: attention space dimension
    :param ln: whether use layer normalization
    :param num_heads: attention head number
    :param dropout: attention dropout, default disable
    :param out_map: output additional mapping
    :param cache: cache-based decoding
    :param count_mask: counting vector for l0drop
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

        if mem_mask is not None:
            logits += mem_mask

        # modifying 'weights = tf.nn.softmax(logits)' to include the counting information.
        # --------
        logits = logits - tf.reduce_max(logits, -1, keepdims=True)
        exp_logits = tf.exp(logits)

        # basically, the count considers how many states are dropped (i.e. gate value 0s)
        if count_mask is not None:
            exp_logits *= count_mask

        exp_sum_logits = tf.reduce_sum(exp_logits, -1, keepdims=True)
        weights = exp_logits / exp_sum_logits
        # --------

        dweights = util.valid_apply_dropout(weights, dropout)

        # weights * v => attention vectors
        o = tf.matmul(dweights, v)
        o = func.combine_heads(o)

        if out_map:
            o = func.linear(o, hidden_size, ln=ln, scope="o_map")

        results = {
            'weights': weights,
            'output': o,
            'cache': cache
        }

        return results


def extract_encodes(source_memory, source_mask, l0_mask):
    x_shp = util.shape_list(source_memory)

    l0_mask = dtype.tf_to_float(tf.cast(l0_mask, tf.bool))
    l0_mask = tf.squeeze(l0_mask, -1) * source_mask

    # count retained encodings
    k_value = tf.cast(tf.reduce_max(tf.reduce_sum(l0_mask, 1)), tf.int32)
    # batch_size x k_value
    _, topk_indices = tf.nn.top_k(l0_mask, k_value)

    # prepare coordinate
    x_pos = util.batch_coordinates(x_shp[0], k_value)
    coord = tf.stack([x_pos, topk_indices], axis=2)

    # gather retained features
    g_x = tf.gather_nd(source_memory, coord)
    g_mask = tf.gather_nd(l0_mask, coord)

    # padding zero
    g_x = tf.pad(g_x, [[0, 0], [1, 0], [0, 0]])

    # generate counts, i.e. how many tokens are dropped
    droped_number = tf.reduce_sum(source_mask, 1) - tf.reduce_sum(l0_mask, 1)
    pad_mask = dtype.tf_to_float(tf.greater(droped_number, 0.))
    droped_number = tf.where(tf.less_equal(droped_number, 0.), tf.ones_like(droped_number), droped_number)

    count_mask = tf.ones_like(g_mask)
    count_mask = tf.concat([tf.expand_dims(droped_number, 1), count_mask], 1)

    g_mask = tf.concat([tf.expand_dims(pad_mask, 1), g_mask], 1)

    return g_x, g_mask, count_mask


def encoder(source, params):
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
                    x = func.layer_norm(x)

                with tf.variable_scope("feed_forward"):
                    y = func.ffn_layer(
                        x,
                        params.filter_size,
                        hidden_size,
                        dropout=params.relu_dropout,
                    )

                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    x = func.layer_norm(x)

    source_encodes = x
    x_shp = util.shape_list(x)

    return {
        "encodes": source_encodes,
        "decoder_initializer": {
            "layer_{}".format(l): {
                "k": dtype.tf_to_float(tf.zeros([x_shp[0], 0, hidden_size])),
                "v": dtype.tf_to_float(tf.zeros([x_shp[0], 0, hidden_size])),
            }
            for l in range(params.num_decoder_layer)
        },
        "mask": mask
    }


def decoder(target, state, params):
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

    # Applying L0Drop
    # --------
    source_memory = state["encodes"]
    source_mask = state["mask"]

    # source_pruning: log alpha_i = x_i w^T
    source_pruning = func.linear(source_memory, 1, scope="source_pruning")

    if is_training:  # training
        source_memory, l0_mask = l0norm.var_train((source_memory, source_pruning))
        l0_norm_loss = tf.squeeze(l0norm.l0_norm(source_pruning), -1)
        l0_norm_loss = tf.reduce_sum(l0_norm_loss * source_mask, -1) / tf.reduce_sum(source_mask, -1)
        l0_norm_loss = tf.reduce_mean(l0_norm_loss)
        l0_norm_loss = l0norm.l0_regularization_loss(
            l0_norm_loss,
            reg_scalar=params.l0_norm_reg_scalar,
            start_reg_ramp_up=params.l0_norm_start_reg_ramp_up,
            end_reg_ramp_up=params.l0_norm_end_reg_ramp_up,
            warm_up=params.l0_norm_warm_up,
        )

        # force the model to only attend to unmasked position
        source_mask = dtype.tf_to_float(tf.cast(tf.squeeze(l0_mask, -1), tf.bool)) * source_mask
    else:  # evaluation
        source_memory, l0_mask = l0norm.var_eval((source_memory, source_pruning))
        l0_norm_loss = 0.0

        source_memory, source_mask, count_mask = extract_encodes(source_memory, source_mask, l0_mask)
        count_mask = tf.expand_dims(tf.expand_dims(count_mask, 1), 1)
    # --------

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
                    x = func.layer_norm(x)

                with tf.variable_scope("cross_attention"):
                    if is_training:
                        y = func.dot_attention(
                            x,
                            source_memory,
                            func.attention_bias(source_mask, "masking"),
                            hidden_size,
                            num_heads=params.num_heads,
                            dropout=params.attention_dropout,
                         )
                    else:
                        y = dot_attention(
                            x,
                            source_memory,
                            func.attention_bias(source_mask, "masking"),
                            hidden_size,
                            count_mask=count_mask,
                            num_heads=params.num_heads,
                            dropout=params.attention_dropout,
                            cache=state['decoder']['state']['layer_{}'.format(layer)]
                        )

                        # mk, mv
                        state['decoder']['state']['layer_{}'.format(layer)] \
                            .update(y['cache'])

                    y = y['output']
                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    x = func.layer_norm(x)

                with tf.variable_scope("feed_forward"):
                    y = func.ffn_layer(
                        x,
                        params.filter_size,
                        hidden_size,
                        dropout=params.relu_dropout,
                    )

                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    x = func.layer_norm(x)
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

    loss = loss + l0_norm_loss

    # these mask tricks mainly used to deal with zero shapes, such as [0, 1]
    loss = tf.cond(tf.equal(tf.shape(target)[0], 0),
                   lambda: tf.constant(0, tf.float32),
                   lambda: loss)

    return loss, logits, state, per_sample_loss


def train_fn(features, params, initializer=None):
    with tf.variable_scope(params.scope_name or "model",
                           initializer=initializer,
                           reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx()),
                           custom_getter=dtype.float32_variable_storage_getter):
        state = encoder(features['source'], params)
        loss, logits, state, _ = decoder(features['target'], state, params)

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
        state = encoder(features['source'], params)
        _, _, _, scores = decoder(features['target'], state, params)

        return {
            "score": scores
        }


def infer_fn(params):
    params = copy.copy(params)
    params = util.closing_dropout(params)

    def encoding_fn(source):
        with tf.variable_scope(params.scope_name or "model",
                               reuse=tf.AUTO_REUSE,
                               dtype=tf.as_dtype(dtype.floatx()),
                               custom_getter=dtype.float32_variable_storage_getter):
            state = encoder(source, params)
            state["decoder"] = {
                "state": state["decoder_initializer"]
            }
            return state

    def decoding_fn(target, state, time):
        with tf.variable_scope(params.scope_name or "model",
                               reuse=tf.AUTO_REUSE,
                               dtype=tf.as_dtype(dtype.floatx()),
                               custom_getter=dtype.float32_variable_storage_getter):
            if params.search_mode == "cache":
                state['time'] = time
                step_loss, step_logits, step_state, _ = decoder(
                    target, state, params)
                del state['time']
            else:
                estate = encoder(state, params)
                estate['dev_decode'] = True
                _, step_logits, _, _ = decoder(target, estate, params)
                step_state = state

            return step_logits, step_state

    return encoding_fn, decoding_fn


# register the model, with a unique name
model.model_register("transformer_l0drop", train_fn, score_fn, infer_fn)
