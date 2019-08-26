# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from func import linear
from rnns import cell as cell


class sru(cell.Cell):
    """The Simple Recurrent Unit."""

    def __init__(self, d, ln=False, scope='sru'):
        super(sru, self).__init__(d, ln=ln, scope=scope)

    def get_init_state(self, shape=None, x=None, scope=None):
        return self._get_init_state(
            self.d * 2, shape=shape, x=x, scope=scope)

    def get_hidden(self, x):
        return tf.split(x, 2, -1)[0]

    def fetch_states(self, x):
        with tf.variable_scope(
                "fetch_state_{}".format(self.scope or "sru")):
            h = linear(x, self.d * 4,
                       bias=False, ln=self.ln, scope="hide_x")
        return (h, )

    def __call__(self, h_, x):
        # h_: the concatenation of previous hidden state
        #    and memory cell state
        # x_r/x: the current input state for r gate
        # x_f/x: the current input state for f gate
        # x_c/x: the current input state for candidate cell
        # x_h/x: the current input state for hidden output
        #   we increase this because we do not assume that
        #   the input dimension equals the output dimension
        """
            f = sigmoid(Wx, vf * c_)
            c = f * c_ + (1 - f) * Wx
            r = sigmoid(Wx, vr * c_)
            h = r * c + (1 - r) * Ux
        """
        if isinstance(x, (list, tuple)):
            x = x[0]

        with tf.variable_scope(
                "cell_{}".format(self.scope or "sru")):
            x_r, x_f, x_c, x_h = tf.split(x, 4, -1)
            h_, c_ = tf.split(h_, 2, -1)

            v_f = tf.get_variable("v_f", [1, self.d])
            v_r = tf.get_variable("v_r", [1, self.d])

            f = tf.sigmoid(x_f + v_f * c_)
            c = f * c_ + (1. - f) * x_c
            r = tf.sigmoid(x_r + v_r * c_)
            h = r * c + (1. - r) * x_h

        return tf.concat([h, c], -1)
