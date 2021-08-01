# coding: utf-8
# author: Biao Zhang

import config
from devices import gpu, tpu
from devices import tpu_feeder_queue, gpu_feeder_queue


# put the graph (model) on the corresponding devices (cpu/gpu/tpu)
def on_device(graph):
  p = config.p()

  if p.use_tpu:
    return tpu.TPUModel(
      graph, 
      p.tpu_name, p.tpu_zone, p.tpu_project,
      iterations_per_loop=p.iterations_per_loop,
      num_hosts=None if p.tpu_num_hosts < 0 else p.tpu_num_hosts,
      num_cores=None if p.tpu_num_cores < 0 else p.tpu_num_cores,
    )
  else:
    return gpu.GPUModel(graph, p.gpus)


# put the cpu dataset on the corresponding device queue
def on_device_queue(device_graph):
  p = config.p()

  if p.use_tpu:
    return tpu_feeder_queue.TPUDataFeedingQueue(
      p.iterations_per_loop > 1,
      device_graph,
      stop_freq=p.tpu_stop_freq * p.update_cycle,
      queue_size=p.tpu_queue_size,
    )
  else:
    return gpu_feeder_queue.GPUDataFeedingQueue(
      p.iterations_per_loop > 1,
      queue_size=p.gpu_queue_size,
    )
