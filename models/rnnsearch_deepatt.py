# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

import func
from models import model
from utils import util, dtype
from rnns import rnn


"""
Reimplementation of TPAMI paper: Neural Machine Translation with Deep Attention
https://ieeexplore.ieee.org/document/8493282

Example Training: 

data=path-to-wmt14-ende/

python run.py --mode train --parameters=hidden_size=1000,embed_size=620,\
dropout=0.1,label_smooth=0.1,attention_dropout=0.1,\
max_len=100,batch_size=25,eval_batch_size=16,\
token_size=5000,batch_or_token='batch',\
initializer="uniform",initializer_gain=0.08,\
model_name="rnnsearch_deepatt",scope_name="nmt",buffer_size=600000,\
clip_grad_norm=0.0,\
num_heads=1,\
process_num=3,\
estop_patience=100,\
num_encoder_layer=5,\
num_decoder_layer=5,\
warmup_steps=4000,\
lrate_strategy="epoch",\
lrate=5e-4,lrate_decay=0.5,cell="gru",\
epoches=5000,\
update_cycle=4,\
gpus=[0],\
disp_freq=1,\
eval_freq=5000,\
sample_freq=1000,\
checkpoints=5,\
max_training_steps=300000,\
nthreads=8,\
beta1=0.9,\
beta2=0.999,\
epsilon=1e-8,\
swap_memory=True,\
layer_norm=True,\
random_seed=1234,\
src_vocab_file="$data/vocab.zero.en",\
tgt_vocab_file="$data/vocab.zero.de",\
src_train_file="$data/train.32k.en.shuf",\
tgt_train_file="$data/train.32k.de.shuf",\
src_dev_file="$data/dev.32k.en",\
tgt_dev_file="$data/dev.32k.de",\
src_test_file="$data/newstest2014.32k.en",\
tgt_test_file="$data/newstest2014.de",\
output_dir="train",\
test_output="trans.txt",
"""


# encoder of deep attention model
def encoder(source, params):
    mask = dtype.tf_to_float(tf.cast(source, tf.bool))
    hidden_size = params.hidden_size

    source, mask = util.remove_invalid_seq(source, mask)

    # extract source word embedding and apply dropout
    embed_name = "embedding" if params.shared_source_target_embedding \
        else "src_embedding"
    src_emb = tf.get_variable(embed_name,
                              [params.src_vocab.size(), params.embed_size])
    src_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(src_emb, source)
    inputs = tf.nn.bias_add(inputs, src_bias)

    inputs = util.valid_apply_dropout(inputs, params.dropout)

    # the encoder module used in the deep attention paper
    with tf.variable_scope("encoder"):
        # x: embedding input, h: the hidden state
        x = inputs
        h = 0
        z = 0

        for layer in range(params.num_encoder_layer+1):
            with tf.variable_scope("layer_{}".format(layer)):
                if layer == 0:
                    # for the first layer, we perform a normal rnn layer to collect context information
                    outputs = rnn.rnn(params.cell, x, hidden_size, mask=mask,
                                      ln=params.layer_norm, sm=params.swap_memory)
                    h = outputs[1][0]

                else:
                    # for deeper encoder layers, we incorporate both embedding input and previous inversed hidden
                    # state sequence as input.
                    # the embedding informs current input while hidden state tells future context
                    is_reverse = (layer % 2 == 1)
                    outputs = rnn.cond_rnn(params.cell, tf.reverse(x, [1]) if is_reverse else x,
                                           tf.reverse(h, [1]) if is_reverse else h, hidden_size,
                                           mask=tf.reverse(mask, [1]) if is_reverse else mask,
                                           ln=params.layer_norm,
                                           sm=params.swap_memory,
                                           num_heads=params.num_heads,
                                           one2one=True)
                    h = outputs[1][0]
                    h = tf.reverse(h, [1]) if is_reverse else h

                # the final hidden state used for decoder state initialization
                z = outputs[1][1]

    with tf.variable_scope("decoder_initializer"):
        decoder_cell = rnn.get_cell(
            params.cell, hidden_size, ln=params.layer_norm
        )

    return {
        "encodes": h,
        "decoder_initializer": {'layer': decoder_cell.get_init_state(x=z, scope="dec_init_state")},
        "mask": mask
    }


# decoding recurent cell for deep attention model
def deep_att_dec_rnn(cell_name, x, memory, d, init_state=None,
             mask=None, mem_mask=None, ln=False, sm=True,
             depth=1, num_heads=1):
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
    # depth: depth for the decoder in deep attention
    # num_heads: number of attention heads, multi-head attention
    # dp: variational dropout

    in_shape = util.shape_list(x)
    batch_size, time_steps = in_shape[:2]
    mem_shape = util.shape_list(memory)

    cell_lower = rnn.get_cell(cell_name, d, ln=ln,
                              scope="{}_lower".format(cell_name))
    cells_higher = []
    for layer in range(depth):
        cell_higher = rnn.get_cell(cell_name, d, ln=ln,
                                   scope="{}_higher_{}".format(cell_name, layer))
        cells_higher.append(cell_higher)

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
    proj_memories = func.linear(memory, mem_shape[-1], bias=False,
                                ln=ln, scope="context_att")

    mask_ta = tf.transpose(tf.expand_dims(mask, -1), [1, 0, 2])
    init_context = dtype.tf_to_float(tf.zeros([batch_size, depth, mem_shape[-1]]))
    init_weight = dtype.tf_to_float(tf.zeros([batch_size, depth, num_heads, mem_shape[1]]))
    mask_pos = len(cache_inputs)

    def _step_fn(prev, x):
        t, h_, c_, a_ = prev

        m, v = x[mask_pos], x[:mask_pos]

        # the first decoder rnn subcell, composing previous hidden state with the current word embedding
        s_ = cell_lower(h_, v)
        s_ = m * s_ + (1. - m) * h_

        atts, att_ctxs = [], []

        for layer in range(depth):
            # perform attention
            prev_cell = cell_lower if layer == 0 else cells_higher[layer-1]
            vle = func.additive_attention(
                prev_cell.get_hidden(s_), memory, mem_mask,
                mem_shape[-1], ln=ln, num_heads=num_heads,
                proj_memory=proj_memories, scope="deep_attention_{}".format(layer))
            a, c = vle['weights'], vle['output']
            atts.append(tf.expand_dims(a, 1))
            att_ctxs.append(tf.expand_dims(c, 1))

            # perform next-level recurrence
            c_c = cells_higher[layer].fetch_states(c)
            ss_ = cells_higher[layer](s_, c_c)
            s_ = m * ss_ + (1. - m) * s_

        h = s_
        a = tf.concat(atts, axis=1)
        c = tf.concat(att_ctxs, axis=1)

        return t + 1, h, c, a

    time = tf.constant(0, dtype=tf.int32, name="time")
    step_states = (time, init_state, init_context, init_weight)
    step_vars = cache_inputs + [mask_ta]

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
    # batch x target length x depth x mem-dimension
    contexts = tf.transpose(context_ta, [1, 0, 2, 3])
    # batch x num_heads x depth x target length x source length
    attentions = tf.transpose(attention_ta, [1, 3, 2, 0, 4])

    return (outputs, output_states), \
           (cells_higher[-1].get_hidden(outputs), cells_higher[-1].get_hidden(output_states)), \
        contexts, attentions


# decoder of deep attention model
def decoder(target, state, params):
    mask = dtype.tf_to_float(tf.cast(target, tf.bool))
    hidden_size = params.hidden_size

    is_training = ('decoder' not in state)

    # handling target-side word embedding, including shift-padding for training
    embed_name = "embedding" if params.shared_source_target_embedding \
        else "tgt_embedding"
    tgt_emb = tf.get_variable(embed_name,
                              [params.tgt_vocab.size(), params.embed_size])
    tgt_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(tgt_emb, target)
    inputs = tf.nn.bias_add(inputs, tgt_bias)

    # shift
    if is_training:
        inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])
        inputs = inputs[:, :-1, :]
    else:
        inputs = tf.cond(tf.reduce_all(tf.equal(target, params.tgt_vocab.pad())),
                         lambda: tf.zeros_like(inputs),
                         lambda: inputs)
        mask = tf.ones_like(mask)

    inputs = util.valid_apply_dropout(inputs, params.dropout)

    with tf.variable_scope("decoder"):
        x = inputs

        init_state = state["decoder_initializer"]["layer"]
        if not is_training:
            init_state = state["decoder"]["state"]["layer"]
        returns = deep_att_dec_rnn(params.cell, x, state["encodes"], hidden_size,
                                   init_state=init_state, mask=mask,
                                   num_heads=params.num_heads,
                                   mem_mask=state["mask"], ln=params.layer_norm,
                                   sm=params.swap_memory, depth=params.num_decoder_layer)
        (_, hidden_state), (outputs, _), contexts, attentions = returns

        if not is_training:
            state['decoder']['state']['layer'] = hidden_state

        x = outputs
        cshp = util.shape_list(contexts)
        c = tf.reshape(contexts, [cshp[0], cshp[1], cshp[2]*cshp[3]])

    feature = func.linear(tf.concat([x, c, inputs], -1), params.embed_size, ln=params.layer_norm, scope="ff")
    feature = tf.nn.tanh(feature)

    feature = util.valid_apply_dropout(feature, params.dropout)

    if 'dev_decode' in state:
        feature = feature[:, -1, :]

    embed_name = "tgt_embedding" if params.shared_target_softmax_embedding \
        else "softmax_embedding"
    embed_name = "embedding" if params.shared_source_target_embedding \
        else embed_name
    softmax_emb = tf.get_variable(embed_name,
                                  [params.tgt_vocab.size(), params.embed_size])
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
                step_loss, step_logits, step_state, _ = decoder(
                    target, state, params)
            else:
                estate = encoder(state, params)
                estate['dev_decode'] = True
                _, step_logits, _, _ = decoder(target, estate, params)
                step_state = state

            return step_logits, step_state

    return encoding_fn, decoding_fn


# register the model, with a unique name
model.model_register("rnnsearch_deepatt", train_fn, score_fn, infer_fn)
