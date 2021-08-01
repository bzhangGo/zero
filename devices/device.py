# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import numpy as np

from tensorflow.python.framework import tensor_spec


# modeling operation that should be run on devices
class ModelOp(
  collections.namedtuple('ModelOp', [
    'compile_op', 'execute_op', 'infeed_op', 'outfeed_op', 'outfeed_mapper'
  ])):
  pass


# abstract device functions
class DeviceModel(object):
  def __init__(self, graph):
    # graph: a model wrapper object
    self._graph = graph
    self._iterations_per_loop = 1

  @abc.abstractmethod
  def get_session(self):
    """Get GPU/TPU running session"""
    raise NotImplementedError("Not Supported")

  @abc.abstractmethod
  def num_device(self):
    """Get the number of devices"""
    raise NotImplementedError("Not Supported")

  @abc.abstractmethod
  def tower_train_graph(self, placeholders, optimizer):
    """Supporting model training, return model ops and output parsers"""
    raise NotImplementedError("Not Supported")

  @abc.abstractmethod
  def tower_score_graph(self, placeholders):
    """Supporting model scoring, return model ops and output parsers"""
    raise NotImplementedError("Not Supported")

  @abc.abstractmethod
  def tower_infer_graph(self, placeholders):
    """Supporting model decoding, return model ops and output parsers"""
    raise NotImplementedError("Not Supported")
  
  @abc.abstractmethod
  def restore_model_postadjust(self):
    """Restore model parameters from checkpoints with postprocess, for TPU in particular"""
    raise NotImplementedError("Not Supported")


class DeviceDataFeedingQueue(object):
  """Device Queue for training"""
  def __init__(self, device_feed):
    self._dataiter = None
    self._skip_steps = -1
    self._device_feed = device_feed

  def on(self, dataiter, skip_steps=-1):
    self._dataiter = dataiter
    self._skip_steps = skip_steps
    return self

  @abc.abstractmethod
  def wakeup(self):
    """Wake up feeding operation"""
    raise NotImplementedError("Not Supported")


class ParsedOutputs(object):
  """A helper function for holding and parsing outputs"""
  def __init__(self, outputs):
    self._outputs = outputs

  def get_output(self):
    return self._outputs

  def aggregate_dictlist_with_key(self, key, reduction='average'):
    assert key in self._outputs[0], "The required `{}` is not in our object".format(key)

    values = [o[key] for o in self._outputs]

    if reduction == 'average':
      return np.mean(values)
    elif reduction == 'sum':
      return np.sum(values)
    elif reduction == 'list':
      return values
    else:
      raise NotImplementedError('No supported reduction {}'.format(reduction))


class DeviceMapper(object):
  """A helper function for manipulating placeholder and model outputs, dict-based"""
  def __init__(self, dict_obj):
    assert isinstance(dict_obj, dict), \
      'Mapper only supporting dict object, not {}'.format(type(dict_obj))

    self.obj = dict_obj

  def get_values(self):
    return [self.obj[k] for k in self.obj]

  def get_specs(self):
    return [tensor_spec.TensorSpec(v.shape, v.dtype, v.name) for v in self.get_values()]

  def get_keys(self):
    return [k for k in self.obj]

  def get_dict(self, values=None):
    if values is None:
      return self.obj
    else:
      assert len(values) == len(self.get_values()), \
        'The given values must have the same shape as the ' \
        'mapper value, but {} != {}'.format(len(values), len(self.get_values()))

      return {k: v for k, v in zip(self.get_keys(), values)}

  def parse(self, outputs, num_devices):
    num_devices = max(num_devices, 1)

    len_o = len(outputs)
    len_d = len(self.get_specs()) * num_devices
    assert len_o == len_d, 'Output & Devices mismatch, {} vs. {}'.format(len_o, len_d)

    parsed_outputs = []
    shard_size = len(self.get_specs())
    for shard_id in range(num_devices):
      shard_output = outputs[shard_id * shard_size: (shard_id + 1) * shard_size]
      parsed_outputs.append(self.get_dict(shard_output))

    return ParsedOutputs(parsed_outputs)

