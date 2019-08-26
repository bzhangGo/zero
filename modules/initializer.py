# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import dtype


def get_initializer(initializer, initializer_gain):
    tfdtype = tf.as_dtype(dtype.floatx())

    if initializer == "uniform":
        max_val = initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val, dtype=tfdtype)
    elif initializer == "normal":
        return tf.random_normal_initializer(0.0, initializer_gain, dtype=tfdtype)
    elif initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal",
                                               dtype=tfdtype)
    elif initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform",
                                               dtype=tfdtype)
    else:
        tf.logging.warn("Unrecognized initializer: %s" % initializer)
        tf.logging.warn("Return to default initializer: glorot_uniform_initializer")
        return tf.glorot_uniform_initializer(dtype=tfdtype)


def scale_initializer(scale, initializer):
    """Rescale the value given by initializer"""
    tfdtype = tf.as_dtype(dtype.floatx())

    def _initializer(shape, dtype=tfdtype, partition_info=None):
        value = initializer(shape, dtype=dtype, partition_info=partition_info)
        value *= scale

        return value

    return _initializer
