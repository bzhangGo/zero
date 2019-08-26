# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Copied from Keras

# the type of float to use throughout the session.
_FLOATX = 'float32'
_EPSILON = 1e-8
_INF = 1e8


def epsilon():
    return _EPSILON


def set_epsilon(e):
    global _EPSILON
    _EPSILON = e


def inf():
    return _INF


def set_inf(e):
    global _INF
    _INF = e


def floatx():
    return _FLOATX


def set_floatx(floatx):
    global _FLOATX
    if floatx not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(floatx))
    _FLOATX = str(floatx)


def np_to_float(x):
    return np.asarray(x, dtype=_FLOATX)


def tf_to_float(x):
    return tf.cast(x, tf.as_dtype(floatx()))


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable
