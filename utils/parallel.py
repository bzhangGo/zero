# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
import tensorflow.contrib as tc

from tensorflow.python.training import device_setter
from tensorflow.python.framework import device as pydev
from tensorflow.core.framework import node_def_pb2

from utils import util, dtype


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops is None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser


def _maybe_repeat(x, n):
    if isinstance(x, list):
        assert len(x) == n
        return x
    else:
        return [x] * n


def _reshape_output(outputs):
    # assumption: or outputs[0] are all tensor lists/tuples,
    #             or outputs[0] are dictionaries
    if isinstance(outputs[0], (tuple, list)):
        outputs = list(zip(*outputs))
        outputs = tuple([list(o) for o in outputs])
    else:
        if not isinstance(outputs[0], dict):
            return outputs

        assert isinstance(outputs[0], dict), \
            'invalid data type %s' % type(outputs[0])

        combine_outputs = {}
        for key in outputs[0]:
            combine_outputs[key] = [o[key] for o in outputs]
        outputs = combine_outputs

    return outputs


# Data-level parallelism
def data_parallelism(device_type, num_devices, fn, *args, **kwargs):
    # Replicate args and kwargs
    if args:
        new_args = [_maybe_repeat(arg, num_devices) for arg in args]
        # Transpose
        new_args = [list(x) for x in zip(*new_args)]
    else:
        new_args = [[] for _ in range(num_devices)]

    new_kwargs = [{} for _ in range(num_devices)]

    for k, v in kwargs.items():
        vals = _maybe_repeat(v, num_devices)

        for i in range(num_devices):
            new_kwargs[i][k] = vals[i]

    fns = _maybe_repeat(fn, num_devices)

    # Now make the parallel call.
    outputs = []
    for i in range(num_devices):
        worker = "/{}:{}".format(device_type, i)
        if device_type == 'cpu':
            _device_setter = local_device_setter(worker_device=worker)
        else:
            _device_setter = local_device_setter(
                ps_device_type='gpu',
                worker_device=worker,
                ps_strategy=tc.training.GreedyLoadBalancingStrategy(
                    num_devices, tc.training.byte_size_load_fn)
            )

        with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0),
                               dtype=tf.as_dtype(dtype.floatx())):
            with tf.name_scope("tower_%d" % i):
                with tf.device(_device_setter):
                    outputs.append(fns[i](*new_args[i], **new_kwargs[i]))

    return _reshape_output(outputs)


def parallel_model(model_fn, features, devices, use_cpu=False):
    device_type = 'gpu'
    num_devices = len(devices)

    if use_cpu:
        device_type = 'cpu'
        num_devices = 1

    outputs = data_parallelism(device_type, num_devices, model_fn, features)

    return outputs


def average_gradients(tower_grads, mask=None):
    """Modified from Bilm"""

    # optimizer for single device
    if len(tower_grads) == 1:
        return tower_grads[0]

    # calculate average gradient for each shared variable across all GPUs
    def _deduplicate_indexed_slices(values, indices):
        """Sums `values` associated with any non-unique `indices`."""
        unique_indices, new_index_positions = tf.unique(indices)
        summed_values = tf.unsorted_segment_sum(
            values, new_index_positions,
            tf.shape(unique_indices)[0])
        return summed_values, unique_indices

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # We need to average the gradients across each GPU.

        g0, v0 = grad_and_vars[0]

        if g0 is None:
            # no gradient for this variable, skip it
            tf.logging.warn("{} has no gradient".format(v0.name))
            average_grads.append((g0, v0))
            continue

        if isinstance(g0, tf.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #   a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            if mask is None:
                avg_values = tf.concat(values, 0) / len(grad_and_vars)
            else:
                avg_values = tf.concat(values, 0) / tf.reduce_sum(mask)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)
        else:
            # a normal tensor can just do a simple average
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            if mask is not None:
                grad = tf.boolean_mask(
                    grad, tf.cast(mask, tf.bool), axis=0)
            grad = tf.reduce_mean(grad, 0)

        # the Variables are redundant because they are shared
        # across towers. So.. just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    assert len(average_grads) == len(list(zip(*tower_grads)))

    return average_grads
