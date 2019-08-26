# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def relative_attention_inner(x, y, z=None, transpose=False):
    """Relative position-aware dot-product attention inner calculation.
    This batches matrix multiply calculations to avoid unnecessary broadcasting.

    Args:
    x: Tensor with shape [batch_size, heads, length, length or depth].
    y: Tensor with shape [batch_size, heads, length, depth].
    z: Tensor with shape [length, length, depth].
    transpose: Whether to transpose inner matrices of y and z. Should be true if
        last dimension of x is depth, not length.

    Returns:
    A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size = tf.shape(x)[0]
    heads = x.get_shape().as_list()[1]
    length = tf.shape(x)[2]

    # xy_matmul is [batch_size, heads, length, length or depth]
    xy_matmul = tf.matmul(x, y, transpose_b=transpose)
    if z is not None:
        # x_t is [length, batch_size, heads, length or depth]
        x_t = tf.transpose(x, [2, 0, 1, 3])
        # x_t_r is [length, batch_size * heads, length or depth]
        x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
        # x_tz_matmul is [length, batch_size * heads, length or depth]
        x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
        # x_tz_matmul_r is [length, batch_size, heads, length or depth]
        x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
        # x_tz_matmul_r_t is [batch_size, heads, length, length or depth]
        x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
        return xy_matmul + x_tz_matmul_r_t
    else:
        return xy_matmul


def get_relative_positions_embeddings(length_x, length_y,
                                      depth, max_relative_position, name=None, last=None):
    """Generates tensor of size [length_x, length_y, depth]."""
    with tf.variable_scope(name or "rpr"):
        relative_positions_matrix = get_relative_positions_matrix(
            length_x, length_y, max_relative_position)
        # to handle cached decoding, where target-token incrementally grows
        if last is not None:
            relative_positions_matrix = relative_positions_matrix[-last:]
        vocab_size = max_relative_position * 2 + 1
        # Generates embedding for each relative position of dimension depth.
        embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
        embeddings = tf.gather(embeddings_table, relative_positions_matrix)
        return embeddings


def get_relative_positions_matrix(length_x, length_y, max_relative_position):
    """Generates matrix of relative positions between inputs."""
    range_vec_x = tf.range(length_x)
    range_vec_y = tf.range(length_y)

    # shape: [length_x, length_y]
    distance_mat = tf.expand_dims(range_vec_x, -1) - tf.expand_dims(range_vec_y, 0)
    distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                            max_relative_position)

    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat
