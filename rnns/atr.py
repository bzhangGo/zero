# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from func import linear
from rnns import cell as cell


class atr(cell.Cell):
    """The Addition-Subtraction Twin-Gated Recurrent Unit."""

    def __init__(self, d, ln=False, twin=True, scope='atr'):
        super(atr, self).__init__(d, ln=ln, scope=scope)

        self.twin = twin

    def get_init_state(self, shape=None, x=None, scope=None):
        return self._get_init_state(
            self.d, shape=shape, x=x, scope=scope)

    def fetch_states(self, x):
        with tf.variable_scope(
                "fetch_state_{}".format(self.scope or "atr")):
            h = linear(x, self.d,
                       bias=False, ln=self.ln, scope="hide_x")
        return (h, )

    def __call__(self, h_, x):
        # h_: the previous hidden state
        # x: the current input state
        """
            p = W x
            q = U h_
            i = sigmoid(p + q)
            f = sigmoid(p - q)
            h = i * p + f * h_
        """
        if isinstance(x, (list, tuple)):
            x = x[0]

        with tf.variable_scope(
                "cell_{}".format(self.scope or "atr")):
            q = linear(h_, self.d,
                       ln=self.ln, scope="hide_h")
            p = x

            f = tf.sigmoid(p - q)
            if self.twin:
                i = tf.sigmoid(p + q)
            # we empirically find that the following simple form is more stable.
            else:
                i = 1. - f

            h = i * p + f * h_

        return h
