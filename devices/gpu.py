# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from devices.device import ModelOp
from devices.device import DeviceModel
from devices.device import DeviceMapper

import six
import atexit
import config
from utils import cycle
from utils import dtype
from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.python.training import device_setter
from tensorflow.python.framework import device as pydev
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.platform import tf_logging as logging


def get_session(gpus):
  """Config session with GPUS"""

  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  if len(gpus) > 0:
    device_str = ",".join([str(i) for i in gpus])
    sess_config.gpu_options.visible_device_list = device_str
  sess = tf.Session(config=sess_config)

  return sess


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops is None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser


class GPUModel(DeviceModel):
  def __init__(self, graph, gpu_devices):
    super(GPUModel, self).__init__(graph)
    self._gpu_devices = gpu_devices

    self._device_type = "gpu"
    if len(gpu_devices) == 0:
      logging.warn("Note you didn't specify any GPU devices,"
                   "backwords to cpu running!")
      self._device_type = "cpu"

    self._session = None

  def get_session(self):
    if self._session is not None:
      return self._session

    self._session = get_session(self._gpu_devices)

    def _shut_down_gpu_system():
      self._session.close()
      logging.info("{}: Goodbye :)".format(self.device_type()))
    atexit.register(_shut_down_gpu_system)

    return self._session

  def num_device(self):
    return max(1, len(self._gpu_devices))

  def device_type(self):
    return self._device_type

  def _tower_graph(self, graph_fn, placeholders, optimizer=None):
    """
    Rough procedure for GPU/CPU model computation:
    1. cpu placeholder -> get GPU placeholder
    2. GPU placeholder -> build GPU models -> GPU outputs
    3. GPU outputs -> cpu outputs

    Compared to TPU operations, GPU doesn't require hardware-level infeed queue and outfeed queue
    """

    features = placeholders['device_placeholder']
    loss_scale = tf.cast(config.p().loss_scale, tf.float32)
    logging.info("Start Compilation")

    shard_outputs = []
    for shard_id in range(self.num_device()):
      worker = "/{}:{}".format(self.device_type(), shard_id)
      if self.device_type() == 'cpu':
        _device_setter = local_device_setter(worker_device=worker)
      else:
        _device_setter = local_device_setter(
          ps_device_type='gpu',
          worker_device=worker,
          ps_strategy=tc.training.GreedyLoadBalancingStrategy(
            self.num_device(), tc.training.byte_size_load_fn)
        )

      gpu_shard_feature = features[shard_id]

      with tf.variable_scope(
        tf.get_variable_scope(), reuse=bool(shard_id != 0),
        dtype=tf.as_dtype(dtype.floatx())):
        with tf.name_scope("tower_%d" % shard_id):
          with tf.device(_device_setter):
            # get output
            output = graph_fn(gpu_shard_feature)

            if optimizer is not None:
              # scale the loss in terms of number of shards
              loss = output["loss"] / self.num_device()

              tvars = tf.trainable_variables()
              grads_and_vars = optimizer.compute_gradients(
                loss*loss_scale,
                tvars,
                colocate_gradients_with_ops=True,
              )
              grads_and_vars = [(g/loss_scale, v) for g, v in grads_and_vars]

              shard_outputs.append([output, grads_and_vars])
            else:
              shard_outputs.append(output)

    output_mapper = None
    ordered_outputs = OrderedDict()
    # Generate GPU training ops when optimizer is available
    if optimizer is not None:
      tower_grads_and_vars = [o[1] for o in shard_outputs]
      tower_outputs = [o[0] for o in shard_outputs]

      # preparing tower-gathered gradients
      tower_grads_and_vars_separate = [list(zip(*gvs)) for gvs in tower_grads_and_vars]
      sep_gradients = [gvs_[0] for gvs_ in tower_grads_and_vars_separate]
      sep_variables = [gvs_[1] for gvs_ in tower_grads_and_vars_separate]

      summed_grads = [tf.add_n(list(gs)) for gs in zip(*sep_gradients)]
      summed_vars = sep_variables[0]

      grads_and_vars = [(g, v) for g, v in zip(summed_grads, summed_vars)]

      # preparing tower-gathered outputs
      for tower_output in tower_outputs:
        output_mapper = DeviceMapper(tower_output)
        for value in output_mapper.get_values():
          ordered_outputs["elements_%d" % len(ordered_outputs)] = value

      named_outputs, train_op = cycle.cycle_optimizer(ordered_outputs, grads_and_vars, optimizer)

      # there are some extra variables added to `named_outputs` compared to `ordered_outputs`
      # we will spread these variables to each device outputs
      extra_outputs = {key: named_outputs[key] for key in named_outputs if "elements_" not in key}
      ordered_named_outputs = OrderedDict()
      for key in ordered_outputs:
        if key in named_outputs:
          ordered_named_outputs[key] = named_outputs[key]
      parsed_results = output_mapper.parse(ordered_named_outputs.values(), self.num_device())

      ordered_outputs = OrderedDict()
      for tower_output in parsed_results.get_output():
        tower_output.update(extra_outputs)
        output_mapper = DeviceMapper(tower_output)
        for value in output_mapper.get_values():
          ordered_outputs["elements_%d" % len(ordered_outputs)] = value

      named_outputs = ordered_outputs
    else:
      # preparing tower-gathered outputs
      for tower_output in shard_outputs:
        output_mapper = DeviceMapper(tower_output)
        for value in output_mapper.get_values():
          ordered_outputs["elements_%d" % len(ordered_outputs)] = value

      named_outputs = ordered_outputs
      train_op = tf.no_op()

    outfeed_op = []
    for key in ordered_outputs:
      outfeed_op.append(tf.identity(named_outputs[key]))
    if optimizer is not None:
      outfeed_op = [outfeed_op]

    assert self._iterations_per_loop == 1, 'Opus, iteration per loop is not supported for GPU!'

    logging.info("End Compilation")

    return ModelOp(
      tf.no_op(),
      train_op,
      infeed_op=tf.no_op(),
      outfeed_op=outfeed_op,
      outfeed_mapper=output_mapper,
    )

  def tower_train_graph(self, placeholders, optimizer):
    return self._tower_graph(self._graph.train_fn, placeholders, optimizer)

  def tower_score_graph(self, placeholders):
    return self._tower_graph(self._graph.score_fn, placeholders)

  def tower_infer_graph(self, placeholders):
    return self._tower_graph(self._graph.infer_fn, placeholders)

  def restore_model_postadjust(self):
    pass
