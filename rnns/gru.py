# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from func import linear
from rnns import cell as cell


class gru(cell.Cell):
    """The Gated Recurrent Unit."""

    def __init__(self, d, ln=False, scope='gru'):
        super(gru, self).__init__(d, ln=ln, scope=scope)

    def get_init_state(self, shape=None, x=None, scope=None):
        return self._get_init_state(
            self.d, shape=shape, x=x, scope=scope)

    def fetch_states(self, x):
        with tf.variable_scope(
                "fetch_state_{}".format(self.scope or "gru")):
            g = linear(x, self.d * 2,
                       bias=False, ln=self.ln, scope="gate_x")
            h = linear(x, self.d,
                       bias=False, ln=self.ln, scope="hide_x")
        return g, h

    def __call__(self, h_, x):
        # h_: the previous hidden state
        # x_g/x: the current input state for gate
        # x_h/x: the current input state for hidden
        """
            z = sigmoid(h_, x)
            r = sigmoid(h_, x)
            h' = tanh(x, r * h_)
            h = z * h_ + (1. - z) * h'
        """
        with tf.variable_scope(
                "cell_{}".format(self.scope or "gru")):
            x_g, x_h = x

            h_g = linear(h_, self.d * 2,
                         ln=self.ln, scope="gate_h")
            z, r = tf.split(
                tf.sigmoid(x_g + h_g), 2, -1)

            h_h = linear(h_ * r, self.d,
                         ln=self.ln, scope="hide_h")
            h = tf.tanh(x_h + h_h)

            h = z * h_ + (1. - z) * h

        return h
