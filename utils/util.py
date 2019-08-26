# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pkgutil
import collections
import numpy as np
import tensorflow as tf

from utils import dtype


def batch_indexer(datasize, batch_size):
    """Just divide the datasize into batched size"""
    dataindex = np.arange(datasize).tolist()

    batchindex = []
    for i in range(datasize // batch_size):
        batchindex.append(dataindex[i * batch_size: (i + 1) * batch_size])
    if datasize % batch_size > 0:
        batchindex.append(dataindex[-(datasize % batch_size):])

    return batchindex


def token_indexer(dataset, token_size):
    """Divide the dataset into token-based batch"""
    # assume dataset format: [(len1, len2, ..., lenN)]
    dataindex = np.arange(len(dataset)).tolist()

    batchindex = []

    _batcher = [0.] * len(dataset[0])
    _counter = 0
    i = 0
    while True:
        if i >= len(dataset): break

        # attempt put this datapoint into batch
        _batcher = [max(max_l, l)
                    for max_l, l in zip(_batcher, dataset[i])]
        _counter += 1
        for l in _batcher:
            if _counter * l >= token_size:
                # when an extreme instance occur, handle it by making a 1-size batch
                if _counter > 1:
                    batchindex.append(dataindex[i-_counter+1: i])
                    i -= 1
                else:
                    batchindex.append(dataindex[i: i+1])

                _counter = 0
                _batcher = [0.] * len(dataset[0])
                break

        i += 1

    _counter = sum([len(slice) for slice in batchindex])
    if _counter != len(dataset):
        batchindex.append(dataindex[_counter:])
    return batchindex


def mask_scale(value, mask, scale=None):
    """Prepared for masked softmax"""
    if scale is None:
        scale = dtype.inf()
    return value + (1. - mask) * (-scale)


def valid_apply_dropout(x, dropout):
    """To check whether the dropout value is valid, apply if valid"""
    if dropout is not None and 0. <= dropout <= 1.:
        return tf.nn.dropout(x, 1. - dropout)
    return x


def label_smooth(labels, vocab_size, factor=0.1):
    """Smooth the gold label distribution"""
    if 0. < factor < 1.:
        n = tf.cast(vocab_size - 1, tf.float32)
        p = 1. - factor
        q = factor / n

        t = tf.one_hot(tf.cast(tf.reshape(labels, [-1]), tf.int32),
                       depth=vocab_size, on_value=p, off_value=q)
        normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))
    else:
        t = tf.one_hot(tf.cast(tf.reshape(labels, [-1]), tf.int32),
                       depth=vocab_size)
        normalizing = 0.

    return t, normalizing


def closing_dropout(params):
    """Removing all dropouts"""
    for k, v in params.values().items():
        if 'dropout' in k:
            setattr(params, k, 0.0)
        # consider closing label smoothing
        if 'label_smoothing' in k:
            setattr(params, k, 0.0)
    return params


def dict_update(d, u):
    """Recursive update dictionary"""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def shape_list(x):
    # Copied from Tensor2Tensor
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def get_shape_invariants(tensor):
    # Copied from Tensor2Tensor
    """Returns the shape of the tensor but sets middle dims to None."""
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None

    return tf.TensorShape(shape)


def merge_neighbor_dims(x, axis=0):
    """Merge neighbor dimension of x, start by axis"""
    if len(x.get_shape().as_list()) < axis + 2:
        return x

    shape = shape_list(x)
    shape[axis] *= shape[axis+1]
    shape.pop(axis+1)
    return tf.reshape(x, shape)


def unmerge_neighbor_dims(x, depth, axis=0):
    """Inverse of merge_neighbor_dims, axis by depth"""
    if len(x.get_shape().as_list()) < axis + 1:
        return x

    shape = shape_list(x)
    width = shape[axis] // depth
    new_shape = shape[:axis] + [depth, width] + shape[axis+1:]
    return tf.reshape(x, new_shape)


def expand_tile_dims(x, depth, axis=1):
    """Expand and Tile x on axis by depth"""
    x = tf.expand_dims(x, axis=axis)
    tile_dims = [1] * x.shape.ndims
    tile_dims[axis] = depth

    return tf.tile(x, tile_dims)


def gumbel_noise(shape, eps=None):
    """Generate gumbel noise shaped by shape"""
    if eps is None:
        eps = dtype.epsilon()

    u = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(u + eps) + eps)


def log_prob_from_logits(logits):
    """Probability from un-nomalized logits"""
    return logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)


def batch_coordinates(batch_size, beam_size):
    """Batch coordinate indices under beam_size"""
    batch_pos = tf.range(batch_size * beam_size) // beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])

    return batch_pos


def variable_printer():
    """Print parameters"""
    all_weights = {v.name: v for v in tf.trainable_variables()}
    total_size = 0

    for v_name in sorted(list(all_weights)):
        v = all_weights[v_name]
        tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                        str(v.shape).ljust(20))
        v_size = np.prod(np.array(v.shape.as_list())).tolist()
        total_size += v_size
    tf.logging.info("Total trainable variables size: %d", total_size)


def uniform_splits(total_size, num_shards):
    """Split the total_size into uniform num_shards lists"""
    size_per_shards = total_size // num_shards
    splits = [size_per_shards] * (num_shards - 1) + \
             [total_size - (num_shards - 1) * size_per_shards]

    return splits


def fetch_valid_ref_files(path):
    """Extracting valid reference files according to MT convention"""
    path = os.path.abspath(path)
    if tf.gfile.Exists(path):
        return [path]

    if not tf.gfile.Exists(path + ".ref0"):
        tf.logging.warn("Invalid Reference Format {}".format(path))
        return None

    num = 0
    files = []
    while True:
        file_path = path + ".ref%s" % num
        if tf.gfile.Exists(file_path):
            files.append(file_path)
        else:
            break
        num += 1
    return files


def get_session(gpus):
    """Config session with GPUS"""

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    if len(gpus) > 0:
        device_str = ",".join([str(i) for i in gpus])
        sess_config.gpu_options.visible_device_list = device_str
    sess = tf.Session(config=sess_config)

    return sess


def flatten_list(values):
    """Flatten a list"""
    return [v for value in values for v in value]


def remove_invalid_seq(sequence, mask):
    """Pick valid sequence elements wrt mask"""
    # sequence: [batch, sequence]
    # mask: [batch, sequence]
    boolean_mask = tf.reduce_sum(mask, axis=0)

    # make sure that there are at least one element in the mask
    first_one = tf.one_hot(0, tf.shape(boolean_mask)[0],
                           dtype=tf.as_dtype(dtype.floatx()))
    boolean_mask = tf.cast(boolean_mask + first_one, tf.bool)

    filtered_seq = tf.boolean_mask(sequence, boolean_mask, axis=1)
    filtered_mask = tf.boolean_mask(mask, boolean_mask, axis=1)
    return filtered_seq, filtered_mask


def time_str(t=None):
    """String format of the time long data"""
    if t is None:
        t = time.time()
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime(t))
    return ts


def dynamic_load_module(module, prefix=None):
    """Load submodules inside a module, mainly used for model loading, not robust!!!"""
    # loading all models under directory `models` dynamically
    if not isinstance(module, str):
        module = module.__path__
    for importer, modname, ispkg in pkgutil.iter_modules(module):
        if prefix is None:
            __import__(modname)
        else:
            __import__("{}.{}".format(prefix, modname))
