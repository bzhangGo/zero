# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import copy
import random
import socket

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

import models
import main as graph
from vocab import Vocab
from utils.recorder import Recorder
from utils import dtype, util

# define global initial parameters
global_params = tc.training.HParams(
    # whether share source and target word embedding
    shared_source_target_embedding=False,
    # whether share target and softmax word embedding
    shared_target_softmax_embedding=True,

    # decoding maximum length: source length + decode_length
    decode_length=50,
    # beam size
    beam_size=4,
    # length penalty during beam search
    decode_alpha=0.6,
    # noise beam search with gumbel
    enable_noise_beam_search=False,
    # beam search temperature, sharp or flat prediction
    beam_search_temperature=1.0,
    # return top elements, not used
    top_beams=1,
    # which version of beam search to use
    # cache or dev
    search_mode="cache",

    # relative position encoding
    max_relative_position=16,

    # lrate decay
    # number of shards
    nstable=4,
    # start, end: learning rate decay parameters used in GNMT+
    lrdecay_start=600000,
    lrdecay_end=1200000,
    # warmup steps: start point for learning rate stop increaing
    warmup_steps=400,
    # select strategy: noam, gnmt+, epoch, score and vanilla
    lrate_strategy="gnmt+",
    # learning decay rate
    lrate_decay=0.5,
    # when using score, patience number of bad score obtained for one decay
    lrate_patience=1,
    # cosine learning rate schedule period
    cosine_period=5000,
    # cosine factor
    cosine_factor=1,

    # early stopping
    estop_patience=100,

    # initialization
    # type of initializer
    initializer="uniform",
    # initializer range control
    initializer_gain=0.08,

    # parameters for rnnsearch
    # encoder and decoder hidden size
    hidden_size=1000,
    # source and target embedding size
    embed_size=620,
    # dropout value
    dropout=0.1,
    relu_dropout=0.1,
    residual_dropout=0.1,
    # label smoothing value
    label_smooth=0.1,
    # model name
    model_name="rnnsearch",
    # scope name
    scope_name="rnnsearch",
    # gru, lstm, sru or atr
    cell="atr",
    # whether use caencoder
    caencoder=True,
    # whether use layer normalization, it will be slow
    layer_norm=False,
    # whether use deep attention mechanism
    use_deep_att=False,
    # notice that when opening the swap memory switch
    # you can train reasonably larger batch on condition
    # that your system will use much more cpu memory
    swap_memory=True,
    # filter size for transformer
    filter_size=2048,
    # attention dropout
    attention_dropout=0.1,
    # the number of encoder layers, valid for deep nmt
    num_encoder_layer=6,
    # the number of decoder layers, valid for deep nmt
    num_decoder_layer=6,
    # the number of attention heads
    num_heads=8,

    # average attention network
    # whether use masked version or cumsum version
    aan_mask=True,
    # whether use ffn in the model
    use_ffn=False,

    # allowed maximum sentence length
    max_len=100,
    eval_max_len=1000000,
    # constant batch size at 'batch' mode for batch-based batching
    batch_size=80,
    # constant token size at 'token' mode for token-based batching
    token_size=3000,
    # token or batch-based data iterator
    batch_or_token='token',
    # batch size for decoding, i.e. number of source sentences decoded at the same time
    eval_batch_size=32,
    # whether shuffle batches during training
    shuffle_batch=True,

    # aan generalization
    strategies=["aan"],

    # whether use multiprocessing deal with data reading, default true
    process_num=1,
    # buffer size controls the number of sentences readed in one time,
    buffer_size=100,
    # a unique queue in multi-thread reading process
    input_queue_size=100,
    output_queue_size=100,

    # source vocabulary
    src_vocab_file="",
    # target vocabulary
    tgt_vocab_file="",
    # source train file
    src_train_file="",
    # target train file
    tgt_train_file="",
    # source development file
    src_dev_file="",
    # target development file
    tgt_dev_file="",
    # source test file
    src_test_file="",
    # target test file
    tgt_test_file="",
    # output directory
    output_dir="",
    # output during testing
    test_output="",
    # pretrained modeling
    pretrained_model="",

    # adam optimizer hyperparameters
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-9,
    # gradient clipping value
    clip_grad_norm=5.0,
    # the gradient norm upper bound, to avoid wired large gradient norm, only works for safe nan mode
    gnorm_upper_bound=1e20,
    # initial learning rate
    lrate=1e-5,
    # minimum learning rate
    min_lrate=0.0,
    # maximum learning rate
    max_lrate=1.0,

    # maximum epochs
    epoches=10,
    # the effective batch size is: batch/token size * update_cycle * num_gpus
    # sequential update cycle
    update_cycle=1,
    # the number of gpus
    gpus=[0],

    # enable safely handle nan
    safe_nan=False,

    # deep nmt prediction style
    dl4mt_redict=True,

    # exponential moving average for stability, disabled by default
    ema_decay=-1.,

    # data leak buffer threshold
    data_leak_ratio=0.5,

    # enable training deep transformer
    deep_transformer_init=False,

    # print information every disp_freq training steps
    disp_freq=100,
    # evaluate on the development file every eval_freq steps
    eval_freq=10000,
    # save the model parameters every save_freq steps
    save_freq=5000,
    # print sample translations every sample_freq steps
    sample_freq=1000,
    # saved checkpoint number
    checkpoints=5,
    best_checkpoints=1,
    # the maximum training steps, program with stop if epochs or max_training_steps is meet
    max_training_steps=1000,

    # number of threads for threaded reading, seems useless
    nthreads=6,
    # random control, not so well for tensorflow.
    random_seed=1234,
    # whether or not train from checkpoint
    train_continue=True,

    # provide interface to modify the default datatype
    default_dtype="float32",
    dtype_epsilon=1e-8,
    dtype_inf=1e8,
    loss_scale=1.0,

    # l0drop related parameters
    l0_norm_reg_scalar=1.0,
    l0_norm_start_reg_ramp_up=0,
    l0_norm_end_reg_ramp_up=10000,
    l0_norm_warm_up=True,
)

flags = tf.flags
flags.DEFINE_string("config", "", "Additional Mergable Parameters")
flags.DEFINE_string("parameters", "", "Command Line Refinable Parameters")
flags.DEFINE_string("ensemble_dirs", "", "Model directory for ensemble")
flags.DEFINE_string("name", "model", "Description of the training process for distinguishing")
flags.DEFINE_string("mode", "train", "train or test or ensemble")


# saving model configuration
def save_parameters(params, output_dir):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    param_name = os.path.join(output_dir, "param.json")
    with tf.gfile.Open(param_name, "w") as writer:
        tf.logging.info("Saving parameters into {}"
                        .format(param_name))
        writer.write(params.to_json())


# load model configuration
def load_parameters(params, output_dir):
    param_name = os.path.join(output_dir, "param.json")
    param_name = os.path.abspath(param_name)

    if tf.gfile.Exists(param_name):
        tf.logging.info("Loading parameters from {}"
                        .format(param_name))
        with tf.gfile.Open(param_name, 'r') as reader:
            json_str = reader.readline()
            params.parse_json(json_str)
    return params


# build training process recorder
def setup_recorder(params):
    recorder = Recorder()
    # This is for early stopping, currently I did not use it
    recorder.bad_counter = 0    # start from 0
    recorder.estop = False

    recorder.lidx = -1      # local data index
    recorder.step = 0       # global step, start from 0
    recorder.epoch = 1      # epoch number, start from 1
    recorder.lrate = params.lrate     # running learning rate
    recorder.history_scores = []
    recorder.valid_script_scores = []

    # trying to load saved recorder
    record_path = os.path.join(params.output_dir, "record.json")
    record_path = os.path.abspath(record_path)
    if tf.gfile.Exists(record_path):
        recorder.load_from_json(record_path)

    params.add_hparam('recorder', recorder)
    return params


# print model configuration
def print_parameters(params):
    tf.logging.info("The Used Configuration:")
    for k, v in params.values().items():
        tf.logging.info("%s\t%s", k.ljust(20), str(v).ljust(20))
    tf.logging.info("")


def main(_):
    # set up logger
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info("Welcome Using Zero :)")

    pid = os.getpid()
    tf.logging.info("Your pid is {0} and use the following command to force kill your running:\n"
                    "'pkill -9 -P {0}; kill -9 {0}'".format(pid))
    # On clusters, this could tell which machine you are running
    tf.logging.info("Your running machine name is {}".format(socket.gethostname()))

    # load registered models
    util.dynamic_load_module(models, prefix="models")

    # dealing with model ensemble
    if flags.FLAGS.mode == "ensemble":
        all_params = []

        # different models are separated by comma ;
        model_dirs = flags.FLAGS.ensemble_dirs.split(";")
        for midx, model_dir in enumerate(model_dirs):
            # parameters from saved model
            params = copy.deepcopy(global_params)

            # priority: command line > saver > default
            params.parse(flags.FLAGS.parameters)
            if os.path.exists(flags.FLAGS.config):
                params.override_from_dict(eval(open(flags.FLAGS.config).read()))
            params = load_parameters(params, model_dir)
            # override
            if os.path.exists(flags.FLAGS.config):
                params.override_from_dict(eval(open(flags.FLAGS.config).read()))
            params.parse(flags.FLAGS.parameters)

            # modify the output directory based on model_dir :)
            params.output_dir = os.path.abspath(model_dir)

            # loading vocabulary
            tf.logging.info("Begin Loading Vocabulary")
            start_time = time.time()
            params.src_vocab = Vocab(params.src_vocab_file)
            params.tgt_vocab = Vocab(params.tgt_vocab_file)
            tf.logging.info("End Loading Vocabulary, Source Vocab Size {}, "
                            "Target Vocab Size {}, within {} seconds"
                            .format(params.src_vocab.size(), params.tgt_vocab.size(),
                                    time.time() - start_time))

            # print parameters
            tf.logging.info("Parameters of {}-th model".format(midx))
            print_parameters(params)

            all_params.append(params)

        graph.ensemble(all_params)

        return "Over"

    params = global_params

    # try loading parameters
    # priority: command line > saver > default
    params.parse(flags.FLAGS.parameters)
    if os.path.exists(flags.FLAGS.config):
        params.override_from_dict(eval(open(flags.FLAGS.config).read()))
    params = load_parameters(params, params.output_dir)
    # override
    if os.path.exists(flags.FLAGS.config):
        params.override_from_dict(eval(open(flags.FLAGS.config).read()))
    params.parse(flags.FLAGS.parameters)

    # set up random seed
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    tf.set_random_seed(params.random_seed)

    # loading vocabulary
    tf.logging.info("Begin Loading Vocabulary")
    start_time = time.time()
    params.src_vocab = Vocab(params.src_vocab_file)
    params.tgt_vocab = Vocab(params.tgt_vocab_file)
    tf.logging.info("End Loading Vocabulary, Source Vocab Size {}, "
                    "Target Vocab Size {}, within {} seconds"
                    .format(params.src_vocab.size(), params.tgt_vocab.size(),
                            time.time() - start_time))

    # print parameters
    print_parameters(params)

    # set up the default datatype
    dtype.set_floatx(params.default_dtype)
    dtype.set_epsilon(params.dtype_epsilon)
    dtype.set_inf(params.dtype_inf)

    mode = flags.FLAGS.mode
    if mode == "train":
        # save parameters
        save_parameters(params, params.output_dir)

        # load the recorder
        params = setup_recorder(params)

        graph.train(params)
    elif mode == "test":
        graph.evaluate(params)
    elif mode == "score":
        graph.scorer(params)
    else:
        tf.logging.error("Invalid mode: {}".format(mode))


if __name__ == '__main__':

    tf.app.run()
