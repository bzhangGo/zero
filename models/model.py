# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import namedtuple


# global models defined in Zero
_total_models = {}


# Model tuple, determining the structure of the NMT model
# Note that model inputs or outputs are shaped into a Map structure
# feature_fn: defining the placeholder function
class NMTModelWrapper(namedtuple("NMTModelTupleWrapper",
                                 ("train_fn", "score_fn", "infer_fn"))):
  pass


# Registering the model such that you could fetch it by the model name
def model_register(model_name, train_fn, score_fn, infer_fn):
  model_name = model_name.lower()

  if model_name in _total_models:
    raise Exception("Conflict Model Name: {}".format(model_name))

  tf.logging.info("Registering model: {}".format(model_name))

  _total_models[model_name] = NMTModelWrapper(
    train_fn=train_fn,
    score_fn=score_fn,
    infer_fn=infer_fn,
  )


def get_model(model_name):
  model_name = model_name.lower()

  if model_name in _total_models:
    return _total_models[model_name]

  raise Exception("No supported model {}".format(model_name))
