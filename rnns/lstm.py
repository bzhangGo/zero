# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from func import linear
from rnns import cell as cell


class lstm(cell.Cell):
    """The Long-Short Term Memory Unit."""

    def __init__(self, d, ln=False, scope='lstm'):
        super(lstm, self).__init__(d, ln=ln, scope=scope)

    def get_init_state(self, shape=None, x=None, scope=None):
        return self._get_init_state(
            self.d * 2, shape=shape, x=x, scope=scope)

    def get_hidden(self, x):
        return tf.split(x, 2, -1)[0]

    def fetch_states(self, x):
        with tf.variable_scope(
                "fetch_state_{}".format(self.scope or "lstm")):
            g = linear(x, self.d * 3,
                       bias=False, ln=self.ln, scope="gate_x")
            c = linear(x, self.d,
                       bias=False, ln=self.ln, scope="hide_x")
        return g, c

    def __call__(self, h_, x):
        # h_: the concatenation of previous hidden state
        #    and memory cell state
        # x_i/x: the current input state for input gate
        # x_f/x: the current input state for forget gate
        # x_o/x: the current input state for output gate
        # x_c/x: the current input state for candidate cell
        """
            f = sigmoid(h_, x)
            i = sigmoid(h_, x)
            o = sigmoid(h_, x)
            c' = tanh(h_, x)
            c = f * c_ + i * c'
            h = o * tanh(c)
        """
        with tf.variable_scope(
                "cell_{}".format(self.scope or "lstm")):
            x_g, x_c = x
            h_, c_ = tf.split(h_, 2, -1)

            h_g = linear(h_, self.d * 3,
                         ln=self.ln, scope="gate_h")
            i, f, o = tf.split(
                tf.sigmoid(x_g + h_g), 3, -1)

            h_c = linear(h_, self.d,
                         ln=self.ln, scope="hide_h")
            h_c = tf.tanh(x_c + h_c)

            c = i * h_c + f * c_

            h = o * tf.tanh(c)

        return tf.concat([h, c], -1)
