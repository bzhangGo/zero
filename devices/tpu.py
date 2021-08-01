# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time
import signal
import atexit
import numpy as np

import config
from utils import cycle
from devices.device import ModelOp
from devices.device import DeviceModel
from devices.device import DeviceMapper

import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.tpu.proto import compilation_result_pb2 as tpu_compilation_result
# for tensorflow 1.15 adaptation, future work
# from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.contrib.tpu.python.tpu import keras_tpu_variables


def setup_tpu_session(cluster_resolver):
  """Construct or return a `tf.Session` connected to the given cluster."""
  master = cluster_resolver.master()

  cluster_spec = cluster_resolver.cluster_spec()
  config = tf.ConfigProto(
    isolate_session_state=True,
    allow_soft_placement=True,
  )

  if cluster_spec:
    config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())

  tpu_session = tf.Session(target=master, config=config)
  tpu_session.run(tpu.initialize_system())
  tpu_session._tpu_initialized = True

  tf.enable_resource_variables()

  return tpu_session


def get_tpu_system_metadata(tpu_cluster_resolver):
  """Retrieves TPU system metadata given a TPUClusterResolver."""
  master = tpu_cluster_resolver.master()

  cluster_spec = tpu_cluster_resolver.cluster_spec()
  cluster_def = cluster_spec.as_cluster_def() if cluster_spec else None
  tpu_system_metadata = (
    tpu_system_metadata_lib._query_tpu_system_metadata(
      master, cluster_def=cluster_def, query_topology=True))

  return tpu_system_metadata


def on_device_training_loop(func):
  # Value for this attribute is from xla.DebugOptions.StepMarkerLocation.
  setattr(func, "step_marker_location", "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP")
  return func


def get_tpu_worker_name(devices):
  # notice the device format:
  # /job:worker/replica:0/task:2/device:TPU:2, TPU,
  # "worker"-job name, unclear to me how to set it; so get it instead
  # "replica:0"-unclear to me either
  # "task:2"-the 2nd host
  # "TPU:2"-the 2nd TPU on the 2nd host :)
  # Walk device list to identify TPU worker for enqueue/dequeue operations.
  worker_re = re.compile('/job:([^/]+)')
  for device in devices:
    if 'TPU:0' in device.name:
      return worker_re.search(device.name).group(1)

  raise Exception('No TPU found on given worker.')


def check_model_compiles(sess, compile_op, feed_dict=None):
  """Verifies that the given TPUModelOp can be compiled via XLA."""
  logging.info('Started compiling')
  start_time = time.time()

  result = sess.run(compile_op, feed_dict=feed_dict)
  proto = tpu_compilation_result.CompilationResultProto()
  proto.ParseFromString(result)
  if proto.status_error_message:
    raise RuntimeError('Compilation failed: {}'.format(
      proto.status_error_message))

  end_time = time.time()
  logging.info('Finished compiling. Time elapsed: %s secs',
               end_time - start_time)


def replicated_scope(num_tasks, num_replicas, num_bound):
  """Variable scope for constructing replicated variables."""

  def _replicated_variable_getter(getter, name, *args, **kwargs):
    """Getter that constructs replicated variables."""
    collections = kwargs.pop("collections", None)
    if collections is None:
      collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    kwargs["collections"] = []

    variables = []
    index = {}
    for t in range(num_tasks):
      for i in range(num_replicas):
        replica_name = "{}/{}_{}".format(name, t, i)
        if t == 0 and i == 0:
          replica_name = name
        if t * num_replicas + i < num_bound:
          with tf.device("/task:{}/device:TPU:{}".format(t, i)):
            v = getter(replica_name, *args, **kwargs)
            variables.append(v)
          index[t * num_replicas + i] = v
    result = keras_tpu_variables.ReplicatedVariable(name, variables)

    g = tf.get_default_graph()
    # If "trainable" is True, next_creator() will add the member variables
    # to the TRAINABLE_VARIABLES collection, so we manually remove
    # them and replace with the MirroredVariable. We can't set
    # "trainable" to False for next_creator() since that causes functions
    # like implicit_gradients to skip those variables.
    if kwargs.get("trainable", True):
      collections.append(tf.GraphKeys.TRAINABLE_VARIABLES)
      l = g.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
      for v in index.values():
        if v in l:
          l.remove(v)

    # Modification: if the variable is on graph, not adding it to the graph again!
    # Basically meaning tf.AUTO_REUSE
    on_graph = False
    for c in collections:
      l = g.get_collection_ref(c)
      for v in l:
        if type(v) == type(result) and v.op.name == result.op.name:
          on_graph = True
          break
      if on_graph: break

    if not on_graph:
      g.add_to_collections(collections, result)

    return result

  return tf.variable_scope(
    tf.get_variable_scope(), custom_getter=_replicated_variable_getter)


def get_tpu_name(tpu_name):
  if tpu_name != "":
    return tpu_name
  else:
    try:
      if 'KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS' in os.environ:
        tpu_name = os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS']
      else:
        tpu_name = os.environ['TPU_NAME']
    except KeyError:
      raise Exception("Must have $TPU_NAME configured")
    return tpu_name


class TPUAssignment(object):
  """This is object holding TPU resources assignment for the concrete model.
  `TPUDistributionStrategy` is responsible to create the instance of
  `TPUAssignment`, so, it can dynamically adjust the `num_cores` to use based on
  model and input batch sizes.
  """

  # meta data: system information, contains number_hosts, num_cores, by default
  # num_hosts, num_cores: user identified, when you want to use subset of devices, 
  # instead of all
  def __init__(self, metadata, num_hosts=None, num_cores=None):
    self._metadata = metadata

    self._topology = tc.tpu.Topology(metadata.topology)

    self._num_hosts = metadata.num_hosts
    self._num_of_cores_per_host = metadata.num_of_cores_per_host
    self._num_cores = metadata.num_cores

    if num_hosts is not None:
      self._num_hosts = min(num_hosts, self._num_hosts)
    if num_cores is not None:
      self._num_cores = min(num_cores, self._num_cores)

    device_coordinates = self._topology.device_coordinates
    used_device_coordinates = device_coordinates[:self._num_hosts]
    used_device_coordinates = used_device_coordinates.reshape([-1, 1, 3])
    self._device_assignment = tc.tpu.DeviceAssignment(
      self._topology, used_device_coordinates[:self._num_cores]
    )

  @property
  def num_hosts(self):
    return self._num_hosts

  @property
  def num_cores(self):
    return self._num_cores

  @property
  def num_of_cores_per_host(self):
    return self._num_of_cores_per_host

  @property
  def topology(self):
    return self._topology

  @property
  def device_assignment(self):
    return self._device_assignment

  @property
  def devices(self):
    return self._metadata.devices


class TPUModel(DeviceModel):
  def __init__(self,
               graph,
               tpu_name,
               tpu_zone,
               tpu_project,
               iterations_per_loop=1,  # for each step, execute `iterations_per_loop` steps
               num_hosts=None,
               num_cores=None):
    super(TPUModel, self).__init__(graph)

    self._cluster_resolver = tc.cluster_resolver.TPUClusterResolver(
      get_tpu_name(tpu_name), tpu_zone, tpu_project
    )
    self._tpu_assignment = TPUAssignment(
      get_tpu_system_metadata(self._cluster_resolver),
      num_hosts=num_hosts,
      num_cores=num_cores,
    )
    self._worker_name = get_tpu_worker_name(self._tpu_assignment.devices)

    self._iterations_per_loop = iterations_per_loop

    self._session = None
    self._exit_op = None

  def get_session(self):
    if self._session is not None:
      return self._session

    self._session = setup_tpu_session(self._cluster_resolver)
    self._exit_op = tpu.shutdown_system()

    # note that, always remember shutting down the tpu handler
    def _no_kill(signum, frame):
      logging.info("signal number {} is ignored!".format(signum))
      logging.info("Sorry, you can't interrupt tpu shutdown process!")

    def _shut_down_tpu_system():
      signal.signal(signal.SIGINT, _no_kill)
      signal.signal(signal.SIGTERM, _no_kill)

      logging.info("Shutting down TPU system")
      self._session.run(self._exit_op)
      self._session.close()
      logging.info("tpu: Goodbye :)")

    atexit.register(_shut_down_tpu_system)

    return self._session

  def num_device(self):
    return self._tpu_assignment.num_cores

  def _get_infeed_ops(self, placeholders):
    infeed_op = []

    for task_id in range(self._tpu_assignment.num_hosts):
      for shard_id in range(self._tpu_assignment.num_of_cores_per_host):
        with tf.device(
          '/job:%s/task:%d/device:CPU:0' % (self._worker_name, task_id)):
          with tf.device('/task:%d/device:TPU:%d' % (task_id, shard_id)):
            plh_id = task_id * self._tpu_assignment.num_of_cores_per_host + shard_id

            if plh_id < self._tpu_assignment.num_cores:
              shard_feature = placeholders[plh_id]

              shard_feature_mapper = DeviceMapper(shard_feature)
              shard_input_specs = shard_feature_mapper.get_specs()

              infeed_op.append(
                tpu_ops.infeed_enqueue_tuple(
                  shard_feature_mapper.get_values(), [spec.shape for spec in shard_input_specs],
                  name='infeed-enqueue-t%d-s%d' % (task_id, shard_id),
                  device_ordinal=shard_id))

    assert len(infeed_op) == self.num_device(), "The device number doesn't match infeed_op!"

    return infeed_op

  def _get_outfeed_ops(self, outfeed_specs):
    # Build output ops.
    outfeed_op = []
    for task_id in range(self._tpu_assignment.num_hosts):
      for shard_id in range(self._tpu_assignment.num_of_cores_per_host):
        with tf.device(
          '/job:%s/task:%d/device:CPU:0' % (self._worker_name, task_id)):
          plh_id = task_id * self._tpu_assignment.num_of_cores_per_host + shard_id

          if plh_id < self._tpu_assignment.num_cores:
            outfeed_op.extend(
              tpu_ops.outfeed_dequeue_tuple(
                dtypes=[spec.dtype for spec in outfeed_specs],
                shapes=[spec.shape for spec in outfeed_specs],
                name='outfeed-dequeue-t%d-s%d' % (task_id, shard_id),
                device_ordinal=shard_id))

    return outfeed_op

  def _tower_graph(self, graph_fn, placeholders, optimizer=None):
    """
    Rough procedure for TPU model computation:
    1. cpu placeholder -> infeed enqueue
    2. infeed dequeue -> get TPU placeholder
    3. TPU placeholder -> build TPU models -> TPU outputs
    4. TPU outputs -> outfeed enqueue
    5. outfeed dequeue -> cpu outputs
    """

    shard_feature = placeholders['device_placeholder']
    shard_feature_mapper = DeviceMapper(shard_feature[0])
    loss_scale = tf.cast(config.p().loss_scale, tf.float32)

    infeed_specs = shard_feature_mapper.get_specs()
    self._output_mapper = None

    def _model_fn(_step):
      # dequeue infeed, get tpu variables
      infeed_tensors = tpu_ops.infeed_dequeue_tuple(
        dtypes=[spec.dtype for spec in infeed_specs],
        shapes=[spec.shape for spec in infeed_specs],
        name='infeed-dequeue')

      # construct modeling feature
      tpu_shard_feature = shard_feature_mapper.get_dict(infeed_tensors)

      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        with tf.name_scope("tower_modeling"):
          with replicated_scope(
            self._tpu_assignment.num_hosts,
            self._tpu_assignment.num_of_cores_per_host,
            self._tpu_assignment.num_cores
          ):
            # get output
            output = graph_fn(tpu_shard_feature)

      if optimizer is not None:
        tpu_optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        tvars = tf.trainable_variables()
        grads_and_vars = tpu_optimizer.compute_gradients(
          output["loss"] * loss_scale,
          tvars,
        )
        grads_and_vars = [(g / loss_scale, v) for g, v in grads_and_vars]

        # send the normal optimizer to cycle operation,
        # handle cross-shard replica with the model itself
        named_outputs, train_op = cycle.cycle_optimizer(output, grads_and_vars, optimizer)

        output_mapper = DeviceMapper(named_outputs)
      else:
        output_mapper = DeviceMapper(output)

        train_op = tf.no_op()
      self._output_mapper = output_mapper

      ops = tf.group(*[
        train_op,
        tpu_ops.outfeed_enqueue_tuple(
          output_mapper.get_values(),
          name='outfeed-enqueue-train')
      ])

      with tf.control_dependencies([ops]):
        return tf.identity(_step + 1)

    @on_device_training_loop
    def tpu_loop():
      # only when training, we enable iteration per loop
      return tc.tpu.repeat(
        self._iterations_per_loop if optimizer is not None else 1,
        _model_fn,
        [0.]
      )

    # Generate out TPU operations using `tpu.split_compile_and_replicate`.
    # `compile_op` can be used to test the TPU model compiles before execution.
    # `execute op` replicates `_model_fn` `num_replicas` times, with each shard
    # running on a different logical core
    compile_op, execute_op = tpu.split_compile_and_replicate(
      tpu_loop,
      inputs=[[] for _ in range(self.num_device())],
      device_assignment=self._tpu_assignment.device_assignment
    )

    # Generate CPU side operations to enqueue features/labels and dequeue
    # outputs from the model call.
    infeed_ops = self._get_infeed_ops(placeholders["device_placeholder"])

    # Generate CPU side operations to dequeue outputs from the model cell
    assert self._output_mapper is not None, 'ERROR: the output mapper is not well-defined!!!'
    outfeed_op = []
    # training => loops; otherwise, one-step running
    if optimizer is not None:
      for _ in range(self._iterations_per_loop):
        with tf.control_dependencies([tf.group(*outfeed_op)]):
          outfeed_op.append(self._get_outfeed_ops(self._output_mapper.get_specs()))
    else:
      outfeed_op = self._get_outfeed_ops(self._output_mapper.get_specs())

    # check the compiling operation
    feed_dict = {}
    if optimizer is not None:
      for gplh in placeholders['global_placeholder']:
        x = placeholders['global_placeholder'][gplh]
        feed_dict[x] = np.zeros(x.get_shape().as_list(), dtype=x.dtype.as_numpy_dtype)
    check_model_compiles(self.get_session(), compile_op, feed_dict=feed_dict)

    return ModelOp(
      compile_op,
      execute_op,
      infeed_op=infeed_ops,
      outfeed_op=outfeed_op,
      outfeed_mapper=self._output_mapper,
    )

  def tower_train_graph(self, placeholders, optimizer):
    return self._tower_graph(self._graph.train_fn, placeholders, optimizer)

  def tower_score_graph(self, placeholders):
    return self._tower_graph(self._graph.score_fn, placeholders)

  def tower_infer_graph(self, placeholders):
    return self._tower_graph(self._graph.infer_fn, placeholders)

  def restore_model_postadjust(self):
    replica_tpu_var_ops = []
    for var in tf.global_variables():
      if isinstance(var, keras_tpu_variables.ReplicatedVariable):
        replica_tpu_var_ops.append((var, var.assign(var.get())))

    # gradually initialize the model, to avoid protobuf size limit or oom
    chunk_ops = []
    chunk_counter = 0
    tf.logging.info('Starting Ops Feeding')
    for var, op in replica_tpu_var_ops:
      chunk_counter += np.prod(var.shape)
      chunk_ops.append(op)

      if chunk_counter > 2 * 1e7:
        tf.logging.info('Chunk handling %s parameters' % chunk_counter)
        self.get_session().run(tf.group(*chunk_ops))
        chunk_ops = []
        chunk_counter = 0

    if chunk_counter > 0:
      self.get_session().run(tf.group(*chunk_ops))
    tf.logging.info('Ending Ops Feeding')
