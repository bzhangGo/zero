# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
import tensorflow.contrib as tc

# define global initial parameters
_GLOBAL_PARAMS = tc.training.HParams(
  # whether share source and target word embedding
  shared_source_target_embedding=False,
  # whether share target and softmax word embedding
  shared_target_softmax_embedding=True,

  # decoding length: source length + decode_length
  decode_length=50,
  # decoding maximum length
  decode_max_length=256,
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

  # relative position encoding, rpr attention model
  max_relative_position=16,

  # learning rate
  # warmup steps: start point for learning rate stop increaing
  warmup_steps=400,
  # select strategy: noam
  lrate_strategy="noam",

  # initialization
  # type of initializer
  initializer="uniform",
  # initializer range control
  initializer_gain=0.08,

  # parameters for transformer
  # encoder and decoder hidden size
  hidden_size=512,
  # source and target embedding size
  embed_size=512,
  # dropout value
  dropout=0.1,
  relu_dropout=0.1,
  residual_dropout=0.1,
  # label smoothing value
  label_smooth=0.1,
  # model name
  model_name="transformer",
  # scope name
  scope_name="transformer",
  # filter size for transformer
  filter_size=2048,
  # attention dropout
  attention_dropout=0.1,
  # the number of encoder layers
  num_encoder_layer=6,
  # the number of decoder layers
  num_decoder_layer=6,
  # the number of attention heads
  num_heads=8,
  # enable training deep transformer
  deep_transformer_init=False,
  # whether enable language-specific modeling (for multilingual translation)
  use_lang_specific_modeling=False,
  # whether apply random online backtransaltion (for zero-shot translation)
  enable_robt=False,
  # whether enable merged attention in Transformer (false by default)
  enable_fuse=False,

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
  # evaluation, maximum allowed sequence length
  eval_max_len=1000000,
  # evaluation token size per batch
  eval_token_size=3000,
  # evaluation token or batch-based data iterator
  eval_batch_or_token='batch',
  # whether shuffle batches during training
  shuffle_batch=True,

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
  # multilingual translation, language vocabulary
  to_lang_vocab_file="",
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
  # tensorboard output directory
  tboard_dir="",
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
  clip_grad_value=-1.0,
  # initial learning rate
  lrate=1e-5,
  # minimum learning rate
  min_lrate=0.0,
  # maximum learning rate
  max_lrate=1.0,

  # cct modeling
  # alpha maximum value for Gaussion noise
  cct_alpha_value=5.0,
  # gating middle dimensional layer
  cct_relu_dim=128,
  # splits of feed-forward layer
  cct_M=4,
  # cct budget limit
  cct_bucket_p=0.3,

  # maximum epochs
  epoches=10,
  # the effective batch size is: batch/token size * update_cycle * num_gpus
  # sequential update cycle
  update_cycle=1,
  # early stopping
  estop_patience=100,

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
  # how many iterations run per running step
  # only useful for TPU training;
  iterations_per_loop=1,
  # suspend tpu input feeding every tpu_stop_freq steps
  tpu_stop_freq=1000,
  # tpu feeding queue size
  tpu_queue_size=1000,
  # gpu feeding queue size
  gpu_queue_size=1000,

  # number of threads for threaded reading, seems useless
  nthreads=6,
  # random control, not so well for tensorflow.
  random_seed=1234,
  # whether or not train from checkpoint
  train_continue=True,
  # data leak buffer threshold
  data_leak_ratio=0.5,

  # whether use TPU training
  use_tpu=False,
  # the number of gpus
  gpus=[0],
  # TPU setup
  tpu_name="",
  tpu_zone="",
  tpu_project="",
  tpu_num_hosts=-1,
  tpu_num_cores=-1,

  # provide interface to modify the default datatype
  default_dtype="float32",
  dtype_epsilon=1e-8,
  dtype_inf=1e8,
  loss_scale=1.0,
)


# interface to get the global parameters
def p():
  global _GLOBAL_PARAMS
  return _GLOBAL_PARAMS


# saving model configuration
def save_parameters(output_dir):
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MkDir(output_dir)

  param_name = os.path.join(output_dir, "param.json")
  with tf.gfile.Open(param_name, "w") as writer:
    tf.logging.info("Saving parameters into {}".format(param_name))
    writer.write(p().to_json())


# load model configuration
def load_parameters(output_dir):
  param_name = os.path.join(output_dir, "param.json")

  if tf.gfile.Exists(param_name):
    tf.logging.info("Loading parameters from {}".format(param_name))
    with tf.gfile.Open(param_name, 'r') as reader:
      json_str = reader.readline()
      # remove invalid parameters
      values_map = json.loads(json_str)
      for key in values_map.keys():
        if not hasattr(p(), key):
          tf.logging.warn("Skipping {} from saved value {}".format(key, values_map[key]))
          del values_map[key]
      p().override_from_dict(values_map)
  return p()


# To save training processes, inspired by Nematus
class Recorder(object):
  def load_from_json(self, file_name):
    tf.logging.info("Loading recoder file from {}".format(file_name))
    record = json.load(tf.gfile.Open(file_name, 'rb'))
    record = dict((key.encode("UTF-8"), value)
                  for (key, value) in record.items())
    self.__dict__.update(record)

  def save_to_json(self, file_name):
    tf.logging.info("Saving recorder file into {}".format(file_name))
    with tf.gfile.Open(file_name, 'wb') as writer:
      writer.write(json.dumps(self.__dict__, indent=2).encode("utf-8"))
      writer.close()


# build training process recorder
def setup_recorder():
  params = p()
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
  if tf.gfile.Exists(record_path):
    recorder.load_from_json(record_path)

  params.add_hparam('recorder', recorder)
  return params


# print model configuration
def print_parameters():
  tf.logging.info("The Used Configuration:")
  for k, v in p().values().items():
    tf.logging.info("%s\t%s", k.ljust(20), str(v).ljust(20))
  tf.logging.info("")
