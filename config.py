# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

import main as graph
from vocab import Vocab
from recorder import Recorder

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

    # parameters for rnnsearch
    # encoder and decoder hidden size
    hidden_size=1000,
    # source and target embedding size
    embed_size=620,
    # dropout value
    dropout=0.1,
    # label smoothing value
    label_smooth=0.1,
    # model name
    model_name="rnnsearch",
    # gru, lstm, sru or atr
    cell="atr",
    # whether use caencoder
    caencoder=True,
    # whether use layer normalization, it will be slow
    layer_norm=False,
    # notice that when opening the swap memory switch
    # you can train reasonably larger batch on condition
    # that your system will use much more cpu memory
    swap_memory=True,

    # allowed maximum sentence length
    max_len=100,
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

    # adam optimizer hyperparameters
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    # gradient clipping value
    clip_grad_norm=5.0,
    # initial learning rate
    lrate=1e-5,

    # maximum epochs
    epoches=10,
    # the effective batch size is: batch/token size * update_cycle
    # sequential update cycle
    update_cycle=1,
    # the number of gpus
    gpus=[0],

    # print information every disp_freq training steps
    disp_freq=100,
    # evaluate on the development file every eval_freq steps
    eval_freq=10000,
    # print sample translations every sample_freq steps
    sample_freq=1000,
    # saved checkpoint number
    checkpoints=5,
    # the maximum training steps, program with stop if epoches or max_training_steps is metted
    max_training_steps=1000,

    # number of threads for threaded reading, seems useless
    nthreads=6,
    # buffer size controls the number of sentences readed in one time,
    buffer_size=100,
    # a unique queue in multi-thread reading process
    max_queue_size=100,
    # random control, not so well for tensorflow.
    random_seed=1234,
)

flags = tf.flags
flags.DEFINE_string("parameters", "", "Additional Mergable Parameters")
flags.DEFINE_string("mode", "train", "train or test")


def save_parameters(params, output_dir):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    param_name = os.path.join(output_dir, "param.json")
    with tf.gfile.Open(param_name, "w") as writer:
        tf.logging.info("Saving parameters into {}"
                        .format(param_name))
        writer.write(params.to_json())


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


def setup_recorder(params):
    recorder = Recorder()
    # This is for early stopping, currectly I did not use it
    recorder.bad_counter = 0
    recorder.estop = False

    recorder.lidx = -1   # local data index
    recorder.step = 0   # global step
    recorder.epoch = 0  # epoch number
    recorder.history_scores = []
    recorder.valid_script_scores = []

    # trying to load saved recorder
    record_path = os.path.join(params.output_dir, "record.json")
    record_path = os.path.abspath(record_path)
    if tf.gfile.Exists(record_path):
        recorder.load_from_json(record_path)

    params.add_hparam('recorder', recorder)
    return params


def main(_):
    # set up logger
    tf.logging.set_verbosity(tf.logging.INFO)

    params = global_params

    # try loading parameters
    # priority: command line > saver > default
    # 1. load latest path to load parameters
    params.parse(flags.FLAGS.parameters)
    params = load_parameters(params, params.output_dir)
    # 2. refine with command line parameters
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

    mode = flags.FLAGS.mode
    if mode == "train":
        # save parameters
        save_parameters(params, params.output_dir)

        # load the recorder
        params = setup_recorder(params)

        graph.train(params)
    elif mode == "test":
        graph.evaluate(params)
    else:
        tf.logging.error("Invalid mode: {}".format(mode))


if __name__ == '__main__':
    tf.app.run()
