# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import random
import socket

import numpy as np
import tensorflow as tf

import models
import config
import main as graph
from vocab import Vocab
from utils import dtype, util


flags = tf.flags
flags.DEFINE_string("config", "", "Additional Mergable Parameters")
flags.DEFINE_string("parameters", "", "Command Line Refinable Parameters")
flags.DEFINE_string("name", "model", "Description of the training process for distinguishing")
flags.DEFINE_string("mode", "train", "train or test or ensemble")
flags.DEFINE_string("logpath", "", "training or evaluation, redirect logging path")


def main(_):
  # set up logger
  tf.logging.set_verbosity(tf.logging.INFO)

  # reset output logging path
  if flags.FLAGS.logpath != "":
    util.redirect_tf_logging_output(flags.FLAGS.logpath)
    tf.logging.info("Redirecting logging into {}".format(flags.FLAGS.logpath))

  tf.logging.info("Welcome Using Zero :)")

  pid = os.getpid()
  tf.logging.info("Your pid is {0} and use the following command to force kill your running:\n"
                  "'pkill -9 -P {0}; kill -9 {0}'".format(pid))
  # On clusters, this could tell which machine you are running
  tf.logging.info("Your running machine name is {}".format(socket.gethostname()))

  # load registered models
  tf.logging.info("Dynamically loading implemented models")
  util.dynamic_load_module(models, prefix="models")

  # get parameters
  p = config.p()

  # try loading parameters
  # priority: command line > saver > default
  p.parse(flags.FLAGS.parameters)
  if tf.gfile.Exists(flags.FLAGS.config):
    p.override_from_dict(eval(tf.gfile.Open(flags.FLAGS.config).read()))
  p = config.load_parameters(p.output_dir)
  # override
  if tf.gfile.Exists(flags.FLAGS.config):
    p.override_from_dict(eval(tf.gfile.Open(flags.FLAGS.config).read()))
  p.parse(flags.FLAGS.parameters)

  # set up random seed
  random.seed(p.random_seed)
  np.random.seed(p.random_seed)
  tf.set_random_seed(p.random_seed)

  # check parameter validness
  if p.iterations_per_loop > 1:
    assert p.iterations_per_loop % p.update_cycle == 0, \
      'Iter_Loop / update_cycle must be integer, but {} vs {}' \
      ''.format(p.iterations_per_loop, p.update_cycle)
    assert p.sample_freq % p.iterations_per_loop == 0, \
      'Sample Freq / Iter_Loop must be integer, but {} vs {}' \
      ''.format(p.sample_freq, p.iterations_per_loop)
    assert p.eval_freq % p.iterations_per_loop == 0, \
      'Evaluation Freq / Iter_Loop must be integer, but {} vs {}' \
      ''.format(p.eval_freq, p.iterations_per_loop)
    assert p.sample_freq == p.tpu_stop_freq, \
      'Sample Freq must be equal to TPU stop freq, but {} vs {}' \
      ''.format(p.sample_freq, p.tpu_stop_freq)
    assert p.eval_freq % p.tpu_stop_freq == 0, \
      'Evaluation Freq / TPU stop freq must be integer, but {} vs {}' \
      ''.format(p.eval_freq, p.tpu_stop_freq)

  if not p.use_tpu:
    assert p.iterations_per_loop == 1, 'Opus, iteration per loop is not supported for GPU!'

  # loading vocabulary
  tf.logging.info("Begin Loading Vocabulary")
  start_time = time.time()
  p.src_vocab = Vocab(p.src_vocab_file)
  p.tgt_vocab = Vocab(p.tgt_vocab_file)
  tf.logging.info(
    "End Loading Vocabulary, Source Vocab Size {}, Target Vocab Size {}, within {} "
    "seconds".format(p.src_vocab.size(), p.tgt_vocab.size(), time.time() - start_time))

  if p.use_lang_specific_modeling:
    tf.logging.info("Begin Loading Language Vocabulary")
    p.to_lang_vocab = Vocab(p.to_lang_vocab_file)
    tf.logging.info(
      "End Loading Language Vocabulary, To Language Vocab Size {}"
      ".".format(p.to_lang_vocab.size()))
  else:
    p.to_lang_vocab = None

  # print parameters
  config.print_parameters()

  # set up the default datatype
  dtype.set_floatx(p.default_dtype)
  dtype.set_epsilon(p.dtype_epsilon)
  dtype.set_inf(p.dtype_inf)

  mode = flags.FLAGS.mode
  if mode == "train":
    # save parameters
    config.save_parameters(p.output_dir)

    # load the recorder
    config.setup_recorder()

    graph.train()
  elif mode == "test":
    graph.evaluate()
  elif mode == "score":
    graph.scorer()
  else:
    tf.logging.error("Invalid mode: {}".format(mode))


if __name__ == '__main__':

  tf.app.run()
