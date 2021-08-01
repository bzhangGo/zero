# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import util, dtype
from collections import namedtuple
from tensorflow.python.util import nest


class BeamSearchState(namedtuple("BeamSearchState",
                                 ("inputs", "state", "finish"))):
  pass


class BeamSearchParam(
  namedtuple("BeamSearchParam",
             ("batch_size", "beam_size", "dec_seq_len", "dec_seq_bound",
              "decoding_fn", "pad_id", "eos_id", "vocab_size", "alpha",
              "enable_noise_beam_search", "beam_search_temperature"))):
  pass


def length_penalty(alpha, length):
  """decoding length penalty"""
  return tf.pow((5.0 + tf.cast(length, tf.float32)) / 6.0, alpha)


def get_max_decode_length(source, decode_length, decode_max_length, use_tpu=False):
  """the upper limit of decoding length, device distinguished"""
  # decode_length: the original plan for decoding is source_length + decode_length
  # decode_max_length: hard upper bound limit

  src_mask = dtype.tf_to_float(tf.cast(source, tf.bool))
  source_length = tf.reduce_sum(src_mask, -1)
  max_target_length = source_length + decode_length
  max_target_length = tf.minimum(max_target_length, decode_max_length)
  max_seq_length = tf.reduce_max(max_target_length)

  if use_tpu:
    # fixing model decoding sequence length
    src_seq_len = util.shape_list(source)[1]
    max_seq_length = int(min(
      src_seq_len + decode_length,
      decode_max_length
    ))

  return max_target_length, max_seq_length


def scaler_handler(x, function):
  """Decoding requires to handle cache states, and manipulate
  beam size, but for scales and other functions, this should be
  specifically handled.

  We scape the scalar part
  """
  x = tf.convert_to_tensor(x)

  if len(util.shape_list(x)) < 1:
    return x

  assert callable(function), "function input should be callable!"
  return function(x)


def beam_state_init(state, bsparam):
  """Initialize the beam search state"""
  # state: the cache from encoder
  # decoding_fn: the decoding function, cache is mainly used for decoding
  # batch_size/beam_size/seq_len: determine the shape of the cache
  # pad_id: for dummy sequence id information
  
  decoding_fn = bsparam.decoding_fn
  batch_size = bsparam.batch_size
  beam_size = bsparam.beam_size
  seq_len = bsparam.dec_seq_len
  pad_id = bsparam.pad_id

  seq_len = tf.cast(seq_len, tf.int32)

  # [batch, beam]
  init_log_probs = tf.constant([[0.] + [tf.float32.min] * (beam_size - 1)], dtype=tf.float32)
  init_log_probs = tf.tile(init_log_probs, [batch_size, 1])
  init_scores = tf.zeros_like(init_log_probs)

  # [batch, beam, seq_len], begin-of-sequence
  init_seq = tf.fill([batch_size, beam_size, seq_len], pad_id)
  init_finish_seq = tf.zeros_like(init_seq)

  # [batch, beam]
  init_finish_scores = tf.fill([batch_size, beam_size], tf.float32.min)
  init_finish_flags = tf.zeros([batch_size, beam_size], tf.bool)

  def _setup_cache(prev_seq, state, time):
    # used to initialize some caches
    # this is because pre-compute these caches is to hard,
    # so let's use one dummy run to obtain them.
    # 1. preparing infeeding states
    flat_prev_seqs = util.merge_neighbor_dims(prev_seq, axis=0)
    flat_prev_state = nest.map_structure(
      lambda y: scaler_handler(
        y,
        lambda x: util.merge_neighbor_dims(x, axis=0),
      ),
      state
    )

    # 2. preforming decoding
    _, step_state = decoding_fn(
      flat_prev_seqs[:, time:time+1], flat_prev_state, time)

    # 3. reshape new states and collect cache
    new_state = nest.map_structure(
      lambda y: scaler_handler(
        y,
        lambda x: util.unmerge_neighbor_dims(x, batch_size, axis=0),
      ),
      step_state
    )
    new_state = util.dict_update(new_state, state)

    return new_state

  model_state = _setup_cache(init_seq, state, 0)

  # put all states into cache
  return BeamSearchState(
    inputs=(init_seq, init_log_probs, init_scores),
    state=model_state,
    finish=(init_finish_seq, init_finish_scores, init_finish_flags)
  )


def while_loop_confition_fn(time, bsstate, bsparam):
  """decoding stop condition"""

  # if the maximum time step is reached, or
  # all samples in one batch satisfy that the worst finished sequence
  # score is not better than the best alive sequence score
  alive_log_probs = bsstate.inputs[1]
  finish_scores = bsstate.finish[1]
  finish_flags = bsstate.finish[2]

  # upper bound of length penality
  max_length_penality = length_penalty(bsparam.alpha, bsparam.dec_seq_bound)
  best_alive_score = alive_log_probs[:, 0] / max_length_penality

  # minimum score among finished sequences alone
  worst_finish_score = tf.reduce_min(
    finish_scores * tf.cast(finish_flags, tf.float32), 1)
  # deal with unfinished instances, which is set to `tf.float32.min`
  unfinish_mask = 1. - tf.cast(tf.reduce_any(finish_flags, 1), tf.float32)
  worst_finish_score += unfinish_mask * tf.float32.min

  # boundary
  bound_is_met = tf.reduce_all(
    tf.greater(worst_finish_score, best_alive_score))

  # length constraint
  length_is_met = tf.reduce_any(
    tf.less(time, tf.cast(bsparam.dec_seq_bound, tf.int32)))

  return tf.logical_and(tf.logical_not(bound_is_met), length_is_met)


def while_loop_body_fn(time, bsstate, bsparam):
  """one expansion step of beam search process"""

  # 1. feed previous predictions, and get the next probabilities
  # generating beam * vocab_size predictions
  prev_seq, prev_log_probs, prev_scores = bsstate.inputs

  flat_prev_seqs = util.merge_neighbor_dims(prev_seq, axis=0)
  flat_prev_state = nest.map_structure(
    lambda y: scaler_handler(
      y,
      lambda x: util.merge_neighbor_dims(x, axis=0),
    ),
    bsstate.state
  )

  # decoding with the give state and inputs: [batch * beam, vocab_size]
  decode_target = util.state_time_slice(flat_prev_seqs, time, 1)
  step_logits, step_state = bsparam.decoding_fn(
    decode_target, flat_prev_state, time
  )

  # add gumbel noise into the logits, simulate gumbel top-k sampling without replacement
  if bsparam.enable_noise_beam_search:
    step_logits += util.gumbel_noise(util.shape_list(step_logits))
  # apply temperature decoding
  step_logits /= bsparam.beam_search_temperature

  # get log-probs from logits
  step_log_probs = util.log_prob_from_logits(step_logits)

  # force decoding, avoiding </s> at the first time step
  eos_mask = tf.cast(tf.equal(tf.range(bsparam.vocab_size), bsparam.eos_id), tf.float32)
  step_log_probs = tf.cond(dtype.tf_to_float(time) < dtype.tf_to_float(1.),
                           lambda: step_log_probs + tf.expand_dims(eos_mask, 0) * - dtype.inf(),
                           lambda: step_log_probs)

  # expand to [batch, beam, vocab_size]
  step_log_probs = util.unmerge_neighbor_dims(
    step_log_probs, bsparam.batch_size, axis=0)
  step_state = nest.map_structure(
    lambda y: scaler_handler(
      y,
      lambda x: util.unmerge_neighbor_dims(x, bsparam.batch_size, axis=0),
    ),
    step_state
  )

  # 2. compute top-k scored next predictions
  # reducing beam * vocab_size to 2 * beam
  # [batch, beam, 1] + [batch, beam, vocab_size]
  curr_log_probs = tf.expand_dims(prev_log_probs, 2) + step_log_probs
  decoding_penality = length_penalty(bsparam.alpha, time+1)
  curr_scores = curr_log_probs / decoding_penality

  # [batch, beam, vocab_size] => [batch, beam * vocab_size]
  curr_flat_scores = util.merge_neighbor_dims(curr_scores, axis=1)
  # => [batch, 2 * beam]
  topk_scores, topk_indices = tf.nn.top_k(
    curr_flat_scores, 2 * bsparam.beam_size)

  # index manipulation, [batch, 2 * beam]
  curr_beam_indices = topk_indices // bsparam.vocab_size
  curr_symbol_indices = topk_indices % bsparam.vocab_size
  curr_coordinates = tf.stack([
    util.batch_coordinates(bsparam.batch_size, 2 * bsparam.beam_size),
    curr_beam_indices
  ], axis=2)

  # extract candidate sequences
  # [batch, 2 * beam, time + 1]
  curr_seq = tf.gather_nd(prev_seq, curr_coordinates)
  curr_seq = util.state_time_insert(
    curr_seq,
    tf.expand_dims(curr_symbol_indices, 2),
    time+1,
    axis=2,
  )

  # 3. handling alive sequences
  # reducing 2 * beam to beam
  # detecting end-of-sentence signals
  curr_fin_flags = tf.equal(curr_symbol_indices, bsparam.eos_id)
  alive_scores = topk_scores + tf.cast(curr_fin_flags, tf.float32) * tf.float32.min

  # [batch, 2 * beam] => [batch, beam]
  alive_scores, alive_indices = tf.nn.top_k(alive_scores, bsparam.beam_size)
  beam_pos_coord = util.batch_coordinates(bsparam.batch_size, bsparam.beam_size)
  alive_coordinates = tf.stack([beam_pos_coord, alive_indices], axis=2)

  alive_seq = tf.gather_nd(curr_seq, alive_coordinates)
  alive_beam_indices = tf.gather_nd(curr_beam_indices, alive_coordinates)

  beam_coordinates = tf.stack([beam_pos_coord, alive_beam_indices], axis=2)
  alive_state = nest.map_structure(
    lambda y: scaler_handler(
      y,
      lambda x: tf.gather_nd(x, beam_coordinates),
    ),
    step_state
  )
  alive_log_probs = alive_scores * decoding_penality

  # 4. handle finished sequences
  # reducing 3 * beam to beam
  prev_fin_seq, prev_fin_scores, prev_fin_flags = bsstate.finish
  # [batch, 2 * beam]
  curr_fin_scores = topk_scores + (1.0 - tf.cast(curr_fin_flags, tf.float32)) * tf.float32.min
  # [batch, 3 * beam]
  fin_flags = tf.concat([prev_fin_flags, curr_fin_flags], axis=1)
  fin_scores = tf.concat([prev_fin_scores, curr_fin_scores], axis=1)
  # [batch, beam]
  fin_scores, fin_indices = tf.nn.top_k(fin_scores, bsparam.beam_size)
  fin_coordinates = tf.stack([beam_pos_coord, fin_indices], axis=2)

  fin_flags = tf.gather_nd(fin_flags, fin_coordinates)
  fin_seq = tf.concat([prev_fin_seq, curr_seq], axis=1)
  fin_seq = tf.gather_nd(fin_seq, fin_coordinates)

  next_state = BeamSearchState(
    inputs=(alive_seq, alive_log_probs, alive_scores),
    state=alive_state,
    finish=(fin_seq, fin_scores, fin_flags)
  )

  return time + 1, next_state


def beam_search(features, encoding_fn, decoding_fn, p):

  beam_size = p.beam_size
  alpha = p.decode_alpha
  eos_id = p.tgt_vocab.eos()
  pad_id = p.tgt_vocab.pad()
  vocab_size = p.tgt_vocab.size()  

  batch_size, src_seq_len = util.shape_list(features["source"])
  
  max_target_length, max_seq_length = get_max_decode_length(
    features["source"],
    p.decode_length,
    p.decode_max_length,
    use_tpu=p.use_tpu,
  )
  
  bsparam = BeamSearchParam(
    batch_size=batch_size,
    beam_size=beam_size,
    dec_seq_len=max_seq_length,
    dec_seq_bound=max_target_length,
    decoding_fn=decoding_fn,
    pad_id=pad_id,
    eos_id=eos_id,
    vocab_size=vocab_size,
    alpha=alpha,
    enable_noise_beam_search=p.enable_noise_beam_search,
    beam_search_temperature=p.beam_search_temperature,
  )

  # firstly, run the encoder to get encoder outputs for cache
  model_state = encoding_fn()
  # insert beam_size dimension
  model_state = nest.map_structure(
    lambda y: scaler_handler(
      y,
      lambda x: util.expand_tile_dims(x, beam_size, axis=1),
    ),
    model_state
  )

  init_bsstate = beam_state_init(
    model_state,
    bsparam
  )

  time = tf.constant(0, tf.int32, name="time")
  # shape_invariants = BeamSearchState(
  #   inputs=(tf.TensorShape([batch_size, beam_size, max_target_length]),
  #           tf.TensorShape([batch_size, beam_size]),
  #           tf.TensorShape([batch_size, beam_size])),
  #   state=nest.map_structure(
  #     lambda x: tf.TensorShape(util.shape_list(x)),
  #     init_bsstate.state
  #   ),
  #   finish=(tf.TensorShape([batch_size, beam_size, max_target_length]),
  #           tf.TensorShape([batch_size, beam_size]),
  #           tf.TensorShape([batch_size, beam_size]))
  # )
  outputs = tf.while_loop(
    lambda t, b: while_loop_confition_fn(t, b, bsparam),
    lambda t, b: while_loop_body_fn(t, b, bsparam),
    [time, init_bsstate],
    shape_invariants=None, # [tf.TensorShape([]), shape_invariants],
    parallel_iterations=32,
    back_prop=False)
  final_state = outputs[1]

  alive_seqs = final_state.inputs[0]
  init_scores = final_state.inputs[2]
  final_seqs = final_state.finish[0]
  final_scores = final_state.finish[1]
  final_flags = final_state.finish[2]

  # for those sequences failed decoding, or no finished sequence
  # their alive_seqs can be of arbitrary length, determined by the maximum target length
  # for TPU based decoding, where the maximum length is set to a large number
  # this become problematic
  # fix it by zeroing
  als_shp = util.shape_list(alive_seqs)
  target_boundary = tf.minimum(tf.cast(max_target_length, tf.int32), als_shp[-1])
  target_boundary = tf.one_hot(target_boundary-1, depth=als_shp[-1], dtype=tf.int32)
  target_boundary_mask = tf.cumsum(target_boundary, axis=1, reverse=True)
  alive_seqs *= tf.expand_dims(target_boundary_mask, 1)

  final_seqs = tf.where(tf.reduce_any(final_flags, 1),
                        final_seqs, alive_seqs)
  final_scores = tf.where(tf.reduce_any(final_flags, 1),
                          final_scores, init_scores)

  return {
    'seq': final_seqs[:, :, 1:],
    'score': final_scores
  }
