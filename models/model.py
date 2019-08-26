# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import namedtuple

# global models defined in Zero
_total_models = {}


class ModelWrapper(namedtuple("ModelTupleWrapper",
                              ("train_fn", "score_fn", "infer_fn"))):
    pass


# you need register your model by your self
def model_register(model_name, train_fn, score_fn, infer_fn):
    model_name = model_name.lower()

    if model_name in _total_models:
        raise Exception("Conflict Model Name: {}".format(model_name))

    tf.logging.info("Registering model: {}".format(model_name))

    _total_models[model_name] = ModelWrapper(
        train_fn=train_fn,
        score_fn=score_fn,
        infer_fn=infer_fn,
    )


def get_model(model_name):
    model_name = model_name.lower()

    if model_name in _total_models:
        return _total_models[model_name]

    raise Exception("No supported model {}".format(model_name))
