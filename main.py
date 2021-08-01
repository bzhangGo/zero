# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import cPickle as pkl
import tensorflow as tf

import evalu
import config
import feeder
from data import Dataset
from models import model
from modules import robt, lrs
from devices import on_device, on_device_queue
from utils import util, queuer, saver, tfwriter


def train():
  p = config.p()

  # status measure
  if p.recorder.epoch >= p.epoches or \
    p.recorder.step >= p.max_training_steps:
    tf.logging.info("Stop condition reached, you have finished training your model.")
    return 0.

  # loading dataset
  tf.logging.info("Begin Loading Training and Dev Dataset")
  start_time = time.time()
  size = p.batch_size if p.batch_or_token == 'batch' else p.token_size
  train_dataset = Dataset(
    p.src_train_file,
    p.tgt_train_file,
    p.src_vocab,
    p.tgt_vocab,
    size=size,
    max_len=p.max_len,
    batch_or_token=p.batch_or_token,
    data_leak_ratio=p.data_leak_ratio,
    use_tpu=p.use_tpu,
    to_lang_vocab=p.to_lang_vocab,
  )
  size = p.eval_batch_size if p.eval_batch_or_token == 'batch' else p.eval_token_size
  dev_dataset = Dataset(
    p.src_dev_file,
    p.src_dev_file,
    p.src_vocab,
    p.src_vocab,
    size=size,
    max_len=p.eval_max_len,
    batch_or_token=p.eval_batch_or_token,
    data_leak_ratio=p.data_leak_ratio,
    use_tpu=p.use_tpu,
    to_lang_vocab=p.to_lang_vocab,
  )
  tf.logging.info("End Loading dataset, within {} seconds".format(time.time() - start_time))

  # Build Graph
  with tf.Graph().as_default():

    # get model graph
    tf.logging.info("fetching the model {}".format(p.model_name))
    graph = model.get_model(p.model_name)

    # push model on device
    tf.logging.info("pushing model toward specified devices")
    device_graph = on_device(graph)
    tf.logging.info("Replicating models to {} devices".format(device_graph.num_device()))

    # set up data feeder
    tf.logging.info("setting up the data feeder")
    train_feeder = feeder.Feeder(device_graph.num_device(), is_train=True)
    eval_feeder = feeder.Feeder(device_graph.num_device(), is_train=False)

    # session info
    sess = device_graph.get_session()

    # global step
    # the position is very important, should behind session due to resource variable
    global_step = tf.train.get_or_create_global_step()

    tf.logging.info("Begining Building Training Graph")
    start_time = time.time()

    # set up optimizer
    train_placeholder = train_feeder.get_placeholders()
    lr = lrs.get_lr(p.lrate, global_step, p)
    optimizer = tf.train.AdamOptimizer(
      lr,
      beta1=p.beta1,
      beta2=p.beta2,
      epsilon=p.epsilon
    )

    # set up training graph
    train_model_op = device_graph.tower_train_graph(train_placeholder, optimizer)
    tf.logging.info("End Building Training Graph, within {} seconds".format(time.time() - start_time))

    tf.logging.info("Begin Building Inferring Graph")
    start_time = time.time()

    # set up infer graph
    eval_model_op = device_graph.tower_infer_graph(eval_feeder.get_placeholders())
    tf.logging.info("End Building Inferring Graph, within {} seconds".format(time.time() - start_time))

    # set up potential robt
    robt_feeder, robt_model_op = None, None
    if p.enable_robt:
      robt_feeder, robt_model_op = robt.get_robt_ops(device_graph)

    # initialize the model
    sess.run(tf.global_variables_initializer())

    # setup tensorboard summary writer
    tboard_writer = tfwriter.TFBoardWriter(sess, p.tboard_dir)

    # log parameters
    util.variable_printer()

    # create saver
    train_saver = saver.Saver(
      checkpoints=p.checkpoints,
      output_dir=p.output_dir,
      best_checkpoints=p.best_checkpoints,
    )

    tf.logging.info("Training")
    cycle_counter = 0
    data_for_shards = []
    cum_tokens = []
    device_feeder_counter = 0

    # restore parameters
    tf.logging.info("Trying restore pretrained parameters")
    train_saver.restore(sess, path=p.pretrained_model)

    tf.logging.info("Trying restore existing parameters")
    train_saver.restore(sess)

    # postprocess device-wise model restore, particular for TPU replica processing
    device_graph.restore_model_postadjust()

    # generate device queue
    device_queue = on_device_queue(device_graph)

    # setup learning rate
    p.lrate = p.recorder.lrate

    start_time = time.time()
    start_epoch = p.recorder.epoch
    for epoch in range(start_epoch, p.epoches + 1):

      p.recorder.epoch = epoch

      tf.logging.info("Training the model for epoch {}".format(epoch))

      train_queue = queuer.EnQueuer(
        train_dataset.batcher(buffer_size=p.buffer_size,
                              shuffle=p.shuffle_batch,
                              train=True),
        lambda x: train_dataset.processor(x),
        worker_processes_num=p.process_num,
        input_queue_size=p.input_queue_size,
        output_queue_size=p.output_queue_size,
      )

      # wrapper for gpu/tpu devices
      train_queue = device_queue.on(
        train_queue,
        skip_steps=-1 if not p.train_continue else p.recorder.lidx
      )

      for lidx, data in enumerate(train_queue):
        if p.train_continue:
          if lidx <= p.recorder.lidx:
            segments = p.recorder.lidx // 5
            if p.recorder.lidx < 5 or lidx % segments == 0:
              tf.logging.info("{} Passing {}-th index according "
                              "to record".format(util.time_str(time.time()), lidx))
            continue

        p.recorder.lidx = lidx

        data_for_shards.append(data)
        # use multiple gpus, and data samples is not enough
        # make sure the data is fully added
        # The actual batch size: batch_size * num_gpus * update_cycle
        if len(data_for_shards) < max(1, device_graph.num_device()):
          continue

        device_feeder_counter += 1

        feed_dicts = train_feeder.feed_placeholders(data_for_shards)
        for shard_data in data_for_shards:
          # collect target tokens
          cum_tokens.append(np.sum(shard_data['target'] > 0))

        # reset data points on gpus
        sample_data = data_for_shards
        data_for_shards = []

        if device_feeder_counter % p.iterations_per_loop != 0:
          continue

        if p.iterations_per_loop > 1:
          _, feed_step_outputs = sess.run([train_model_op.execute_op, train_model_op.outfeed_op])
        else:
          _, _, feed_step_outputs = sess.run(
            [train_model_op.infeed_op, train_model_op.execute_op, train_model_op.outfeed_op],
            feed_dict=feed_dicts
          )

        gstep, lrate = sess.run([global_step, lr])

        for loop_iter in range(p.iterations_per_loop):
          last_loop = (loop_iter == p.iterations_per_loop - 1)

          step_outputs = feed_step_outputs[loop_iter]

          # random online backtranslation
          if p.enable_robt:
            assert p.iterations_per_loop == 1, 'RoBT only works with single loop training'

            robt_data = robt.backtranslate(
              sess, sample_data, robt_feeder, robt_model_op)
            feed_dicts = train_feeder.feed_placeholders(robt_data)

            _, _, step_outputs = sess.run([
              train_model_op.infeed_op,
              train_model_op.execute_op,
              train_model_op.outfeed_op,
            ], feed_dict=feed_dicts)

            step_outputs = step_outputs[0]

          cycle_counter += 1

          # at the final step, update model parameters
          if cycle_counter == p.update_cycle:
            cycle_counter = 0

            # handle model outputs
            parsed_outputs = train_model_op.outfeed_mapper.parse(
              step_outputs, device_graph.num_device()
            )

            loss = parsed_outputs.aggregate_dictlist_with_key(
              "loss", reduction="average"
            )
            gnorm = parsed_outputs.aggregate_dictlist_with_key(
              "gradient_norm", reduction="average"
            )
            pnorm = parsed_outputs.aggregate_dictlist_with_key(
              "parameter_norm", reduction="average"
            )

            pgstep = gstep - p.iterations_per_loop // p.update_cycle \
                     + loop_iter // p.update_cycle + 1

            if np.isnan(loss) or np.isinf(loss) or np.isnan(gnorm) or np.isinf(gnorm):
              tf.logging.error("Nan or Inf raised! Loss {} GNorm {}.".format(loss, gnorm))
              # p.recorder.estop = True
              # break
              tf.logging.warn("According to design, this step skipped!")
              continue

            if pgstep % p.disp_freq == 0:
              end_time = time.time()
              tf.logging.info(
                "{} Epoch {}, GStep {}~{}, LStep {}~{}, Loss {:.3f}, GNorm {:.3f}, PNorm {:.3f}, "
                "Lr {:.5f}, Src {}, Tgt {}, Tokens {}, UD {:.3f} s".format(
                  util.time_str(end_time), epoch, pgstep - p.disp_freq + 1, pgstep,
                  lidx - p.disp_freq + 1, lidx, loss, gnorm, pnorm,
                  lrate, data['source'].shape, data['target'].shape,
                  np.sum(cum_tokens) / (p.iterations_per_loop // p.update_cycle), end_time - start_time)
              )
              # printing out to tensorboard
              tboard_writer.scalar("grad_norm", gnorm, pgstep)
              tboard_writer.scalar("loss", loss, pgstep)
              tboard_writer.scalar("lr", lrate, pgstep)
              tboard_writer.scalar("param_norm", pnorm, pgstep)

              start_time = time.time()
              cum_tokens = []

            # trigger model saver
            if last_loop and gstep > 0 and gstep % p.save_freq == 0:
              train_saver.save(sess, gstep)
              p.recorder.save_to_json(os.path.join(p.output_dir, "record.json"))

            # trigger model evaluation
            if last_loop and gstep > 0 and gstep % p.eval_freq == 0:
              tf.logging.info("Start Evaluating")
              eval_start_time = time.time()
              tranes, scores = evalu.decoding(
                sess, eval_feeder, eval_model_op, dev_dataset, p)
              bleu = evalu.eval_metric(tranes, p.tgt_dev_file)
              eval_end_time = time.time()
              tf.logging.info("End Evaluating")

              tf.logging.info(
                "{} GStep {}, Scores {}, BLEU {}, Duration {:.3f} s".format(
                  util.time_str(eval_end_time), gstep, np.mean(scores),
                  bleu, eval_end_time - eval_start_time)
              )
              tboard_writer.scalar("dev_bleu", bleu, gstep)

              # save eval translation
              evalu.dump_tanslation(
                tranes,
                os.path.join(p.output_dir, "eval-{}.trans.txt".format(gstep))
              )

              # save parameters
              train_saver.save(sess, gstep, bleu)

              # check for early stopping
              valid_scores = [v[1] for v in p.recorder.valid_script_scores]
              if len(valid_scores) == 0 or bleu > np.max(valid_scores):
                p.recorder.bad_counter = 0
              else:
                p.recorder.bad_counter += 1

                if p.recorder.bad_counter > p.estop_patience:
                  p.recorder.estop = True
                  break

              p.recorder.history_scores.append((gstep, float(np.mean(scores))))
              p.recorder.valid_script_scores.append((gstep, float(bleu)))
              p.recorder.save_to_json(os.path.join(p.output_dir, "record.json"))

            # trigger temporary sampling
            if last_loop and gstep > 0 and gstep % p.sample_freq == 0:
              tf.logging.info("Start Sampling")

              parsed_decode_outputs = evalu.decode_one_batch(
                sess,
                eval_feeder,
                eval_model_op,
                sample_data,
                batch_size=p.eval_batch_size if p.use_tpu else 5,
                seq_len=p.eval_max_len if p.use_tpu else int(1e8),
                use_tpu=p.use_tpu,
              )

              data = sample_data[0]
              tranes, scores = evalu.decode_hypothesis(parsed_decode_outputs, p)
              for sidx in range(min(5, len(data['source']), len(scores))):
                sample_source = evalu.decode_target_token(data['source'][sidx], p.src_vocab)
                tf.logging.info("{}-th Source: {}".format(sidx, ' '.join(sample_source)))
                sample_target = evalu.decode_target_token(data['target'][sidx], p.tgt_vocab)
                tf.logging.info("{}-th Target: {}".format(sidx, ' '.join(sample_target)))
                sample_trans = tranes[sidx]
                tf.logging.info("{}-th Translation: {}".format(sidx, ' '.join(sample_trans)))

              tf.logging.info("End Sampling")

            # should be equal to global_step
            p.recorder.step = gstep

            # trigger stopping
            if last_loop and gstep >= p.max_training_steps:
              # stop running by setting EStop signal
              p.recorder.estop = True
              break

        if not p.recorder.estop:
          if (gstep > 0 and (gstep % p.eval_freq == 0 or gstep % p.sample_freq == 0)) or p.enable_robt:
            train_queue.wakeup()

        # end for-data loop
        if p.recorder.estop:
          tf.logging.info("Data Early Stopped!")
          break

      # end for-epoch loop
      if p.recorder.estop:
        tf.logging.info("Epoch Early Stopped!")
        break

      # reset to 0
      p.recorder.lidx = -1

  # # Final Evaluation
  # tf.logging.info("Start Final Evaluating")
  #
  # gstep = int(p.recorder.step + 1)
  # eval_start_time = time.time()
  # tranes, scores = evalu.decoding(
  #   sess, eval_feeder, eval_model_op, dev_dataset, p)
  # bleu = evalu.eval_metric(tranes, p.tgt_dev_file)
  # eval_end_time = time.time()
  # tf.logging.info("End Evaluating")
  #
  # tf.logging.info(
  #   "{} GStep {}, Scores {}, BLEU {}, Duration {:.3f} s".format(
  #     util.time_str(eval_end_time), gstep,
  #     np.mean(scores), bleu, eval_end_time - eval_start_time)
  # )
  # tboard_writer.scalar("dev_bleu", bleu, gstep)
  #
  # # save eval translation
  # evalu.dump_tanslation(
  #   tranes,
  #   os.path.join(p.output_dir, "eval-{}.trans.txt".format(gstep))
  # )

  # closing writer resource
  tboard_writer.close()
  tf.logging.info("Your training is finished :)")

  return train_saver.best_score


def evaluate():
  p = config.p()

  # loading dataset
  tf.logging.info("Begin Loading Test Dataset")
  start_time = time.time()
  size = p.eval_batch_size if p.eval_batch_or_token == 'batch' else p.eval_token_size
  test_dataset = Dataset(
    p.src_test_file,
    p.src_test_file,
    p.src_vocab,
    p.src_vocab,
    size=size,
    max_len=p.eval_max_len,
    batch_or_token=p.eval_batch_or_token,
    data_leak_ratio=p.data_leak_ratio,
    use_tpu=p.use_tpu,
    to_lang_vocab=p.to_lang_vocab,
  )

  tf.logging.info(
    "End Loading dataset, within {} seconds".format(time.time() - start_time))

  # Build Graph
  with tf.Graph().as_default():

    # get model graph
    tf.logging.info("fetching the model {}".format(p.model_name))
    graph = model.get_model(p.model_name)

    # push model on device
    tf.logging.info("pushing model toward specified devices")
    device_graph = on_device(graph)

    # set up data feeder
    tf.logging.info("setting up the data feeder")
    eval_feeder = feeder.Feeder(device_graph.num_device(), is_train=False)

    # session info
    sess = device_graph.get_session()

    tf.logging.info("Begining Building Evaluation Graph")
    start_time = time.time()

    # set up infer graph
    eval_model_op = device_graph.tower_infer_graph(eval_feeder.get_placeholders())
    tf.logging.info("End Building Inferring Graph, within {} seconds".format(time.time() - start_time))

    # initialize the model
    sess.run(tf.global_variables_initializer())

    # log parameters
    util.variable_printer()

    # create saver
    eval_saver = saver.Saver(checkpoints=p.checkpoints, output_dir=p.output_dir)

    # restore parameters
    tf.logging.info("Trying restore existing parameters")
    eval_saver.restore(sess, p.output_dir)

    # postprocess device-wise model restore, particular for TPU replica processing
    device_graph.restore_model_postadjust()

    tf.logging.info("Starting Evaluating")
    eval_start_time = time.time()
    tranes, scores = evalu.decoding(
      sess, eval_feeder, eval_model_op, test_dataset, p)
    bleu = evalu.eval_metric(tranes, p.tgt_test_file)
    eval_end_time = time.time()

    tf.logging.info(
      "{} Scores {}, BLEU {}, Duration {}s".format(
        util.time_str(eval_end_time), np.mean(scores), bleu, eval_end_time - eval_start_time)
    )

    # save translation
    evalu.dump_tanslation(tranes, p.test_output)

  return bleu


def scorer():
  p = config.p()

  # loading dataset
  tf.logging.info("Begin Loading Test Dataset")
  start_time = time.time()
  size = p.eval_batch_size if p.eval_batch_or_token == 'batch' else p.eval_token_size
  test_dataset = Dataset(
    p.src_test_file,
    p.tgt_test_file,
    p.src_vocab,
    p.tgt_vocab,
    size=size,
    max_len=p.eval_max_len,
    batch_or_token=p.eval_batch_or_token,
    data_leak_ratio=p.data_leak_ratio,
    use_tpu=p.use_tpu,
    to_lang_vocab=p.to_lang_vocab,
  )
  tf.logging.info(
    "End Loading dataset, within {} seconds".format(time.time() - start_time))

  # Build Graph
  with tf.Graph().as_default():

    # get model graph
    tf.logging.info("fetching the model {}".format(p.model_name))
    graph = model.get_model(p.model_name)

    # push model on device
    tf.logging.info("pushing model toward specified devices")
    device_graph = on_device(graph)

    # set up data feeder
    tf.logging.info("setting up the data feeder")
    score_feeder = feeder.Feeder(device_graph.num_device(), is_train=False, is_score=True)

    # session info
    sess = device_graph.get_session()

    tf.logging.info("Begining Building Evaluation Graph")
    start_time = time.time()

    # set up infer graph
    score_model_op = device_graph.tower_score_graph(score_feeder.get_placeholders())

    tf.logging.info("End Building Scoring Graph, within {} seconds".format(time.time() - start_time))

    # initialize the model
    sess.run(tf.global_variables_initializer())

    # log parameters
    util.variable_printer()

    # create saver
    eval_saver = saver.Saver(checkpoints=p.checkpoints, output_dir=p.output_dir)

    # restore parameters
    tf.logging.info("Trying restore existing parameters")
    eval_saver.restore(sess, p.output_dir)

    # postprocess device-wise model restore, particular for TPU replica processing
    device_graph.restore_model_postadjust()

    tf.logging.info("Starting Evaluating")
    eval_start_time = time.time()
    scores, ppl = evalu.scoring(sess, score_feeder, score_model_op, test_dataset, p)
    eval_end_time = time.time()

    tf.logging.info(
      "{} Scores {}, PPL {}, Duration {}s".format(
        util.time_str(eval_end_time), np.mean(scores['score']), ppl, eval_end_time - eval_start_time)
    )

    # save translation
    evalu.dump_tanslation(scores['score'], p.test_output)
    average_score = np.mean(scores['score'])

    # save other scoring information
    scores = [dict(zip(scores, t)) for t in zip(*scores.values())]
    with open(p.test_output+'.gate.pkl', 'wb') as writer:
      pkl.dump(scores, writer)

  return average_score
