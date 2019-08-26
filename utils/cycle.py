# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import dtype


def _zero_variables(variables, name=None):
    ops = []

    for var in variables:
        with tf.device(var.device):
            op = var.assign(tf.zeros_like(var))
        ops.append(op)

    return tf.group(*ops, name=name or "zero_variables")


def _replicate_variables(variables, device=None, suffix="Replica"):
    new_vars = []

    for var in variables:
        device = device or var.device
        with tf.device(device):
            name = var.op.name + "/{}".format(suffix)
            new_vars.append(tf.Variable(tf.zeros_like(var),
                                        name=name, trainable=False))

    return new_vars


def _collect_gradients(gradients, variables):
    ops = []

    for grad, var in zip(gradients, variables):
        if isinstance(grad, tf.Tensor):
            ops.append(tf.assign_add(var, grad))
        else:
            ops.append(tf.scatter_add(var, grad.indices, grad.values))

    return tf.group(*ops, name="collect_gradients")


def create_train_op(named_scalars, grads_and_vars, optimizer, global_step, params):
    tf.get_variable_scope().set_dtype(tf.as_dtype(dtype.floatx()))

    gradients = [item[0] for item in grads_and_vars]
    variables = [item[1] for item in grads_and_vars]

    if params.update_cycle == 1:
        zero_variables_op = tf.no_op("zero_variables")
        collect_op = tf.no_op("collect_op")
    else:
        named_vars = {}
        for name in named_scalars:
            named_var = tf.Variable(tf.zeros([], dtype=tf.float32),
                                    name="{}/CTrainOpReplica".format(name),
                                    trainable=False)
            named_vars[name] = named_var
        count_var = tf.Variable(tf.zeros([], dtype=tf.as_dtype(dtype.floatx())),
                                name="count/CTrainOpReplica",
                                trainable=False)
        slot_variables = _replicate_variables(variables, suffix='CTrainOpReplica')
        zero_variables_op = _zero_variables(
            slot_variables + [count_var] + list(named_vars.values()))

        collect_ops = []
        # collect gradients
        collect_grads_op = _collect_gradients(gradients, slot_variables)
        collect_ops.append(collect_grads_op)

        # collect other scalars
        for name in named_scalars:
            scalar = named_scalars[name]
            named_var = named_vars[name]
            collect_op = tf.assign_add(named_var, scalar)
            collect_ops.append(collect_op)
        # collect counting variable
        collect_count_op = tf.assign_add(count_var, 1.0)
        collect_ops.append(collect_count_op)

        collect_op = tf.group(*collect_ops, name="collect_op")
        scale = 1.0 / (tf.cast(count_var, tf.float32) + 1.0)
        gradients = [scale * (g + s)
                     for (g, s) in zip(gradients, slot_variables)]

        for name in named_scalars:
            named_scalars[name] = scale * (
                    named_scalars[name] + named_vars[name])

    grand_norm = tf.global_norm(gradients)
    param_norm = tf.global_norm(variables)

    # Gradient clipping
    if isinstance(params.clip_grad_norm or None, float):
        gradients, _ = tf.clip_by_global_norm(gradients,
                                              params.clip_grad_norm,
                                              use_norm=grand_norm)

    # Update variables
    grads_and_vars = list(zip(gradients, variables))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    ops = {
        "zero_op": zero_variables_op,
        "collect_op": collect_op,
        "train_op": train_op
    }

    # apply ema
    if params.ema_decay > 0.:
        tf.logging.info('Using Exp Moving Average to train the model with decay {}.'.format(params.ema_decay))
        ema = tf.train.ExponentialMovingAverage(decay=params.ema_decay, num_updates=global_step)
        ema_op = ema.apply(variables)
        with tf.control_dependencies([ops['train_op']]):
            ops['train_op'] = tf.group(ema_op)
        bck_vars = _replicate_variables(variables, suffix="CTrainOpBackUpReplica")

        ops['ema_backup_op'] = tf.group(*(tf.assign(bck, var.read_value())
                                        for bck, var in zip(bck_vars, variables)))
        ops['ema_restore_op'] = tf.group(*(tf.assign(var, bck.read_value())
                                         for bck, var in zip(bck_vars, variables)))
        ops['ema_assign_op'] = tf.group(*(tf.assign(var, ema.average(var).read_value())
                                        for var in variables))

    ret = named_scalars
    ret.update({
        "gradient_norm": grand_norm,
        "parameter_norm": param_norm,
    })

    return ret, ops
