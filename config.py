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
    shared_source_target_embedding=False,
    shared_target_softmax_embedding=True,

    decode_length=50,
    beam_size=4,
    top_beams=1,
    decode_alpha=0.6,

    # parameters for rnnsearch
    hidden_size=1000,
    embed_size=620,
    dropout=0.1,
    label_smooth=0.1,
    model_name="rnnsearch",
    # gru, lstm or atr
    cell="gru",
    # whether use caencoder
    caencoder=False,
    layer_norm=True,
    # notice that when opening the swap memory switch
    # you can train reasonably larger batch on condition
    # that your system will use much more cpu memory
    swap_memory=True,

    max_len=100,
    batch_size=80,
    token_size=3000,
    # token or batch-based data iterator
    batch_or_token='token',
    eval_batch_size=32,
    shuffle_batch=True,

    src_vocab_file="",
    tgt_vocab_file="",
    src_train_file="",
    tgt_train_file="",
    src_dev_file="",
    tgt_dev_file="",
    src_test_file="",
    tgt_test_file="",
    output_dir="",
    test_output="",

    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    clip_grad_norm=5.0,
    lrate=1e-5,

    epoches=10,
    update_cycle=1,
    gpus=[0],

    disp_freq=100,
    eval_freq=10000,
    sample_freq=1000,
    checkpoints=5,
    max_training_steps=1000,

    nthreads=6,
    buffer_size=100,
    max_queue_size=100,
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

    tf.logging.info("I don't know how to say it, in short, "
                    "programing with tensorflow for rnn has many pits "
                    "to implement high-performance rnn model, "
                    "remember to remove all possible computations "
                    "inside the cell out. In addition, no `tf.while_loop`, "
                    "use `tf.scan` instead. Code will be clean. "
                    "That's a lesson from me !@@!")

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
