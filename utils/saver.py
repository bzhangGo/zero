# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict

import config
from utils import util


class TPU2LocalSaver(object):
  """Fetch variables from TPU-side to Local Disk

  Lession: move tf operation builders outside data generation loop
  TODO: improve the ugly Saver design :-(
  """
  def __init__(self,
               variables,
               checkpoints=5,  # save the latest number of checkpoints
               output_dir=None,  # the output directory
               best_score=-1,  # the best bleu score before
               best_checkpoints=1,  # the best checkpoints saved in best checkpoints directory
               ):
    self.checkpoints = checkpoints
    self.output_dir = output_dir
    self.best_score = best_score
    self.best_checkpoints = best_checkpoints

    self._variables = variables
    self._graph = tf.Graph()

    self._saver, self._sess, self._assign_ops, self._placeholders = self._build_graph()

  def _build_graph(self):
    with self._graph.as_default():
      assign_ops = []
      placeholders = OrderedDict()
      for _var in self._variables:
        name = _var.op.name
        var = tf.get_variable(name, shape=_var.shape, dtype=_var.dtype)
        placeholder = tf.placeholder(var.dtype, shape=var.shape)
        assign_ops.append(tf.assign(var, placeholder))
        placeholders[name] = placeholder

      local_saver = Saver(
        self.checkpoints, self.output_dir, self.best_score, self.best_checkpoints, enable_tpu_saver=False)
      local_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, isolate_session_state=True))
      local_sess.run(tf.global_variables_initializer())

      return local_saver, local_sess, assign_ops, placeholders

  def save(self, named_var_values, step, path, save_to_best=False):
    with self._graph.as_default():
      feed_dict_list = []
      for name in named_var_values:
        feed_dict_list.append([self._placeholders[name], named_var_values[name]])

      Saver.large_ops_with_feeddict(self._sess, self._assign_ops, feed_dict_list)

      local_saver = self._saver.best_saver if save_to_best else self._saver.saver
      local_saver.save(self._sess, path, global_step=step)


class Saver(object):
  """Manipulating model saving and loading

  Support saving/loading from/to local disk and google cloud storage, regardless of
  TPU/GPU/CPU running;

  The latest `checkpoints` checkpoints are saved under `output_dir`; at the same time
  the best `best_checkpoints` checkpoints are saved under `output_dir`/best

  By default, we keep 5 latest checkpoints and 1 best checkpoint. The 5 latest ones are
  often used for checkpoint averaging for model evaluation.
  """
  def __init__(self,
               checkpoints=5,    # save the latest number of checkpoints
               output_dir=None,  # the output directory
               best_score=-1,    # the best bleu score before
               best_checkpoints=1,      # the best checkpoints saved in best checkpoints directory
               enable_tpu_saver=True,   # allow constructing TPU saver: TPU->local disk
               ):
    if output_dir is None:
      output_dir = "./output"
    self.output_dir = output_dir
    self.output_best_dir = os.path.join(output_dir, "best")

    self.saver = tf.train.Saver(
      max_to_keep=checkpoints
    )
    # handle disrupted checkpoints
    if tf.gfile.Exists(self.output_dir):
      ckpt = tf.train.get_checkpoint_state(self.output_dir)
      if ckpt and ckpt.all_model_checkpoint_paths:
        self.saver.recover_last_checkpoints(list(ckpt.all_model_checkpoint_paths))

    self.best_saver = tf.train.Saver(
      max_to_keep=best_checkpoints,
    )
    # handle disrupted checkpoints
    if tf.gfile.Exists(self.output_best_dir):
      ckpt = tf.train.get_checkpoint_state(self.output_best_dir)
      if ckpt and ckpt.all_model_checkpoint_paths:
        self.best_saver.recover_last_checkpoints(list(ckpt.all_model_checkpoint_paths))

    self.best_score = best_score
    # check best bleu result
    metric_dir = os.path.join(self.output_best_dir, "metric.log")
    if tf.gfile.Exists(metric_dir):
      metric_lines = tf.gfile.Open(metric_dir).readlines()
      if len(metric_lines) > 0:
        best_score_line = metric_lines[-1]
        self.best_score = float(best_score_line.strip().split()[-1])

    # check the top_k_best list and results
    self.topk_scores = []
    topk_dir = os.path.join(self.output_best_dir, "topk_checkpoint")
    ckpt_dir = os.path.join(self.output_best_dir, "checkpoint")
    # direct load the topk information from topk_checkpoints
    if tf.gfile.Exists(topk_dir):
      with tf.gfile.Open(topk_dir) as reader:
        for line in reader:
          model_name, score = line.strip().split("\t")
          self.topk_scores.append((model_name, float(score)))
    # backup plan to normal checkpoints and best scores
    elif tf.gfile.Exists(ckpt_dir):
      latest_checkpoint = tf.gfile.Open(ckpt_dir).readline()
      model_name = latest_checkpoint.strip().split(":")[1].strip()
      model_name = model_name[1:-1]  # remove ""
      self.topk_scores.append((model_name, self.best_score))
    self.best_checkpoints = best_checkpoints

    self.metric_dir = metric_dir
    self.score_record = None
    self.local_tpu_saver = None

    if enable_tpu_saver:
      p = config.p()

      if p.use_tpu and not self.output_dir.startswith("gs://"):
        tf.logging.warn("TPU local saver required, back to specific TPU solution")
        self.local_tpu_saver = TPU2LocalSaver(
          tf.global_variables(),
          checkpoints,
          output_dir,
          best_score,
          best_checkpoints,
        )

  @staticmethod
  def large_ops_with_feeddict(session, ops, feed_dict_list):
    # gradually initialize the model, to avoid protobuf size limit
    chunk_ops = []
    chunk_feed_dict = {}
    chunk_counter = 0
    tf.logging.info('Starting Ops Feeding')
    for op, fd_pair in zip(ops, feed_dict_list):
      chunk_counter += np.prod(fd_pair[1].shape)
      chunk_ops.append(op)
      chunk_feed_dict[fd_pair[0]] = fd_pair[1]

      if chunk_counter > 2 * 1e8:
        tf.logging.info('Chunk handling %s parameters' % chunk_counter)
        session.run(tf.group(*chunk_ops), chunk_feed_dict)
        chunk_ops = []
        chunk_feed_dict = {}
        chunk_counter = 0

    if chunk_counter > 0:
      session.run(tf.group(*chunk_ops), chunk_feed_dict)
    tf.logging.info('Ending Ops Feeding')

  def _device_save(self, saver, session, path, step, save_to_best=False):
    p = config.p()

    if not p.use_tpu:
      saver.save(session, path, global_step=step)
    else:
      # when using tpu modeling, saving models to gs or local is an issue
      # by default, tpu models can only be saved from/to google storage
      # this is not reasonable to me. here, I use some trick to handle this
      if path.startswith("gs://"):
        saver.save(session, path, global_step=step)
      else:
        tf.logging.warn("TPU model saving to local checkpoints")
        tf.logging.info("Backupping to specific solutions")

        # saving to local directory, rebuilt the whole variable set
        # step 1. pull out the savable variables
        # split loading into several chunks, avoid protobuf issue
        var_values = []
        chunk_var_list = []
        chunk_counter = 0
        for var in tf.global_variables():
          var_size = np.prod(np.array(var.shape.as_list())).tolist()
          chunk_counter += var_size
          chunk_var_list.append(var)

          if chunk_counter > 2 * 1e8:
            var_values.extend(session.run(chunk_var_list))
            chunk_var_list = []
            chunk_counter = 0

        if chunk_counter > 0:
          var_values.extend(session.run(chunk_var_list))

        assert len(var_values) == len(tf.global_variables()), \
          'The TPU graph variable should be equal to the local variable value'

        named_var_values = OrderedDict()
        for value, var in zip(var_values, tf.global_variables()):
          named_var_values[var.op.name] = value

        # step 2. rebuild graph and variable sets
        assert self.local_tpu_saver is not None, 'You need initialize TPU saver'
        self.local_tpu_saver.save(named_var_values, step, path, save_to_best=save_to_best)

  def save(self, session, step, metric_score=None):
    if not tf.gfile.Exists(self.output_dir):
      tf.gfile.MkDir(self.output_dir)
    if not tf.gfile.Exists(self.output_best_dir):
      tf.gfile.MkDir(self.output_best_dir)

    self._device_save(
      self.saver, session, os.path.join(self.output_dir, "model"), step, save_to_best=False)

    def _move(path, new_path):
      if tf.gfile.Exists(path):
        if tf.gfile.Exists(new_path):
          tf.gfile.Remove(new_path)
        tf.gfile.Copy(path, new_path)

    if metric_score is not None and metric_score > self.best_score:
      self.best_score = metric_score

      _move(os.path.join(self.output_dir, "param.json"),
            os.path.join(self.output_best_dir, "param.json"))
      _move(os.path.join(self.output_dir, "record.json"),
            os.path.join(self.output_best_dir, "record.json"))

      # this recorder only record best scores
      if self.score_record is None:
        util.file_create_if_not_exist(self.metric_dir)
        self.score_record = tf.gfile.Open(self.metric_dir, mode="a+")

      self.score_record.write("Steps {}, Metric Score {}\n".format(step, metric_score))
      self.score_record.flush()

    # either no model is saved, or current metric score is better than the minimum one
    if metric_score is not None and \
      (len(self.topk_scores) == 0 or len(self.topk_scores) < self.best_checkpoints or
       metric_score > min([v[1] for v in self.topk_scores])):
      # manipulate the 'checkpoints', and change the orders
      ckpt_dir = os.path.join(self.output_best_dir, "checkpoint")
      if len(self.topk_scores) > 0:
        sorted_topk_scores = sorted(self.topk_scores, key=lambda x: x[1])
        with tf.gfile.Open(ckpt_dir, mode='w') as writer:
          best_ckpt = sorted_topk_scores[-1]
          writer.write("model_checkpoint_path: \"{}\"\n".format(best_ckpt[0]))
          for model_name, _ in sorted_topk_scores:
            writer.write("all_model_checkpoint_paths: \"{}\"\n".format(model_name))
          writer.flush()

        # update best_saver internal checkpoints status
        ckpt = tf.train.get_checkpoint_state(self.output_best_dir)
        if ckpt and ckpt.all_model_checkpoint_paths:
          self.best_saver.recover_last_checkpoints(list(ckpt.all_model_checkpoint_paths))

      # this change mainly inspired by that sometimes for dataset,
      # the best performance is achieved by averaging top-k checkpoints
      self._device_save(
        self.best_saver, session, os.path.join(self.output_best_dir, "model"), step, save_to_best=True
      )

      # handle topk scores
      self.topk_scores.append(("model-{}".format(int(step)), float(metric_score)))
      sorted_topk_scores = sorted(self.topk_scores, key=lambda x: x[1])
      self.topk_scores = sorted_topk_scores[-self.best_checkpoints:]
      topk_dir = os.path.join(self.output_best_dir, "topk_checkpoint")
      with tf.gfile.Open(topk_dir, mode='w') as writer:
        for model_name, score in self.topk_scores:
          writer.write("{}\t{}\n".format(model_name, score))
        writer.flush()

  def restore(self, session, path=None):
    if path is not None:
      if tf.gfile.Exists(path):
        check_dir = path
      else:
        tf.logging.warn("No checkpoints found at {}! Pass restore!!".format(path))
        return None
    else:
      check_dir = self.output_dir

    checkpoint = os.path.join(check_dir, "checkpoint")
    if not tf.gfile.Exists(checkpoint):
      tf.logging.warn("No Existing Model detected")
    else:
      latest_checkpoint = tf.gfile.Open(checkpoint).readline()
      model_name = latest_checkpoint.strip().split(":")[1].strip()
      model_name = model_name[1:-1]  # remove ""
      model_path = os.path.join(check_dir, model_name)
      if not tf.gfile.Exists(model_path+".meta"):
        tf.logging.error(
          "model '{}' does not exists".format(model_path))
      else:
        try:
          self.saver.restore(session, model_path)
        except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError):
          # In this case, we simply assume that the cycle part
          #   is mismatched, where the replicas are missing.
          # This would happen if you switch from un-cycle mode
          #   to cycle mode.
          # when data in the feed_dict is very large, protobuf size issue appears again.
          #   Solve it by gradually initialize the model, chunk by chunk
          tf.logging.warn("Starting Backup Restore")
          ops = []
          feed_dict_list = []
          reader = tf.train.load_checkpoint(model_path)
          for var in tf.global_variables():
            name = var.op.name

            if reader.has_tensor(name):
              tf.logging.info('{} get initialization from {}'.format(name, name))

              # the reason to use placeholder here is to avoid protobuf size limit
              placeholder = tf.placeholder(var.dtype, shape=var.shape)
              ops.append(tf.assign(var, placeholder))
              feed_dict_list.append([placeholder, reader.get_tensor(name)])
            else:
              tf.logging.warn("{} is missed".format(name))

          # gradually initialize the model, to avoid protobuf size limit
          Saver.large_ops_with_feeddict(session, ops, feed_dict_list)
