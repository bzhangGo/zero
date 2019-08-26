# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
from func import linear
from utils import dtype


# This is an abstract class that deals with
#   recurrent cells, e.g. GRU, LSTM, ATR
class Cell(object):
    def __init__(self,
                 d,             # hidden state dimension
                 ln=False,      # whether use layer normalization
                 scope=None,    # the name scope for this cell
                 ):
        self.d = d
        self.scope = scope
        self.ln = ln

    def _get_init_state(self, d, shape=None, x=None, scope=None):
        # gen init state vector
        # if no evidence x is provided, use zero initialization
        if x is None:
            assert shape is not None, "you should provide shape"
            if not isinstance(shape, (tuple, list)):
                shape = [shape]
            shape = shape + [d]
            return dtype.tf_to_float(tf.zeros(shape))
        else:
            return linear(
                x, d, bias=True, ln=self.ln,
                scope="{}_init".format(scope or self.scope)
            )

    def get_hidden(self, x):
        return x

    @abc.abstractmethod
    def get_init_state(self, shape=None, x=None, scope=None):
        raise NotImplementedError("Not Supported")

    @abc.abstractmethod
    def __call__(self, h_, x):
        raise NotImplementedError("Not Supported")

    @abc.abstractmethod
    def fetch_states(self, x):
        raise NotImplementedError("Not Supported")
