# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import keras_tpu_variables

import config
from utils import dtype


def get_tpu_gradients(grads):
  summed_grads = []

  for grad in grads:
    with ops.colocate_with(grad):
      summed_grads.append(tpu_ops.cross_replica_sum(grad))

  return summed_grads


def cycle_optimizer(named_scalars, grads_and_vars, optimizer):
  # note here, optimizer must be cpu/normal optimizer, **NOT** tpu optimizer
  tf.get_variable_scope().set_dtype(tf.as_dtype(dtype.floatx()))
  tf.get_variable_scope()._reuse = tf.AUTO_REUSE

  global_step = tf.train.get_or_create_global_step()
  p = config.p()

  update_cycle = p.update_cycle

  if update_cycle > 1:
    # define local loop's variables
    local_step = tf.get_variable(name="local_step", shape=[], dtype=tf.int32, trainable=False,
                                 initializer=tf.zeros_initializer())
    batch_finite = tf.get_variable(name="batch_finite", shape=[], dtype=tf.bool, trainable=False,
                                   initializer=tf.ones_initializer())

    # return variable information
    global_norm = tf.get_variable(name="global_norm", shape=[], dtype=tf.float32, trainable=False,
                                  initializer=tf.zeros_initializer())
    named_vars = {}
    for name in named_scalars:
      named_var = tf.get_variable(name="{}/CTrainOpReplica".format(name), shape=[], dtype=tf.float32,
                                  trainable=False, initializer=tf.zeros_initializer())
      named_vars[name] = named_var

    accum_vars = [tf.get_variable(
      name=v.op.name + "/CTrainOpReplica",
      shape=v.shape.as_list(),
      dtype=tf.float32,
      trainable=False,
      initializer=tf.zeros_initializer()) for _, v in grads_and_vars]

    # whether reset information
    reset_step = tf.cast(tf.math.equal(local_step % update_cycle, 0), dtype=tf.bool)
    local_step = tf.cond(reset_step,
                         lambda: local_step.assign(tf.ones_like(local_step)),
                         lambda: local_step.assign_add(1),
                         name='local_step')

    grads = [item[0] for item in grads_and_vars]
    vars = [item[1] for item in grads_and_vars]

    if p.use_tpu:
      grads = get_tpu_gradients(grads)

    # check gradient information, inf/nan or so on
    all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads])
    batch_finite = tf.cond(reset_step,
                           lambda: batch_finite.assign(
                             tf.math.logical_and(tf.constant(True, dtype=tf.bool), all_are_finite)),
                           lambda: batch_finite.assign(
                             tf.math.logical_and(batch_finite, all_are_finite)),
                           name='batch_finite')

    # update accumulated gradients, either reset or adding up
    accum_vars = tf.cond(reset_step,
                         lambda: [var.assign(grad) for var, grad in zip(accum_vars, grads)],
                         lambda: [var.assign_add(grad) for var, grad in zip(accum_vars, grads)],
                         name='accum_vars')

    # collect other scalars
    for name in named_scalars:
      scalar = named_scalars[name]
      named_var = named_vars[name]

      scalar /= update_cycle

      named_var = tf.cond(reset_step,
                          lambda: named_var.assign(scalar),
                          lambda: named_var.assign_add(scalar),
                          name='named_var_%s'%name)
      named_vars[name] = named_var

    def update(accum_vars):
      # 1. average gradients
      grads = [v / update_cycle for v in accum_vars]

      # gradient clipping value
      if isinstance(p.clip_grad_value or None, float) and p.clip_grad_value > 0.0:
        l = - p.clip_grad_value
        h = p.clip_grad_value
        grads = [tf.clip_by_value(g, clip_value_min=l, clip_value_max=h) for g in grads]

      grad_norm = tf.global_norm(grads)

      # 2. gradient clipping
      if isinstance(p.clip_grad_norm or None, float) and p.clip_grad_norm > 0.0:
        grads, _ = tf.clip_by_global_norm(
          grads, clip_norm=p.clip_grad_norm,
          use_norm=tf.cond(batch_finite,
                           lambda: grad_norm, lambda: tf.constant(1.0), name='grad_use_norm_c'))

      _train_op = tf.cond(batch_finite,
                          lambda: optimizer.apply_gradients(
                            list(zip(grads, vars)), global_step=global_step),
                          lambda: tf.no_op(), name="train_op_finite")

      return tf.group(global_norm.assign(grad_norm), _train_op)

    update_step = tf.identity(
      tf.cast(tf.math.equal(local_step % update_cycle, 0), dtype=tf.bool), name="update_step")
    train_op = tf.cond(update_step,
                       lambda: update(accum_vars), lambda: tf.no_op(), name='train_op')
  else:

    grads = [item[0] for item in grads_and_vars]
    vars = [item[1] for item in grads_and_vars]

    if p.use_tpu:
      grads = get_tpu_gradients(grads)

    # check finite, and obtain gradient norm
    all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads])

    # gradient clipping value
    if isinstance(p.clip_grad_value or None, float) and p.clip_grad_value > 0.0:
      l = - p.clip_grad_value
      h =   p.clip_grad_value
      grads = [tf.clip_by_value(g, clip_value_min=l, clip_value_max=h) for g in grads]

    grad_norm = tf.global_norm(grads)

    # gradient clipping norm
    if isinstance(p.clip_grad_norm or None, float) and p.clip_grad_norm > 0.0:
      grads, _ = tf.clip_by_global_norm(
        grads, clip_norm=p.clip_grad_norm,
        use_norm=tf.cond(all_are_finite,
                         lambda: grad_norm, lambda: tf.constant(1.0), name='grad_use_norm_s'))

    train_op = tf.cond(all_are_finite,
                       lambda: optimizer.apply_gradients(
                         list(zip(grads, vars)), global_step=global_step),
                       lambda: tf.no_op(), name="train_op_finite")

    global_norm = grad_norm
    named_vars = copy.copy(named_scalars)

  ret = named_vars
  ret.update({
    "gradient_norm": global_norm,
    "parameter_norm": tf.global_norm([tf.convert_to_tensor(v) for v in vars]),
  })

  return ret, train_op
