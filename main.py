# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

from models import rnnsearch as graph
import evalu
from data import Dataset
from search import beam_search
from utils import parallel, cycle, util, queuer, saver


def tower_train_graph(train_features, optimizer, params):
    # define multi-gpu training graph
    def _tower_train_graph(features):
        loss = graph.train_fn(
            features, params,
            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        tower_gradients = optimizer.compute_gradients(
            loss, colocate_gradients_with_ops=True)
        return loss, tower_gradients

    # feed model to multiple gpus
    tower_outputs, tower_mask = parallel.parallel_model(
        _tower_train_graph, train_features,
        params.gpus, use_cpu=(len(params.gpus) == 0))
    tower_losses, tower_grads = tower_outputs[0], tower_outputs[1]
    loss = tf.reduce_sum(tf.stack(tower_losses, 0) * tower_mask)
    loss /= tf.reduce_sum(tower_mask)
    gradients = parallel.average_gradients(tower_grads, mask=tower_mask)

    return loss, gradients


def tower_infer_graph(eval_features, params):
    # define multi-gpu inferring graph
    def _tower_infer_graph(features):
        encoding_fn, decoding_fn = graph.infer_fn(params)
        seqs, scores = beam_search(features, encoding_fn,
                                   decoding_fn, params)
        return seqs, scores

    # feed model to multiple gpus
    eval_outputs, eval_mask = parallel.parallel_model(
        _tower_infer_graph, eval_features,
        params.gpus, use_cpu=(len(params.gpus) == 0))
    eval_seqs, eval_scores = eval_outputs[0], eval_outputs[1]

    return eval_seqs, eval_scores, eval_mask


def train(params):
    # status measure
    if params.recorder.estop or \
            params.recorder.epoch > params.epoches or \
            params.recorder.step > params.max_training_steps:
        tf.logging.info("Stop condition reached, you have finished training your model.")
        return 0.

    # loading dataset
    tf.logging.info("Begin Loading Training and Dev Dataset")
    start_time = time.time()
    train_dataset = Dataset(params.src_train_file, params.tgt_train_file,
                            params.src_vocab, params.tgt_vocab, params.max_len,
                            batch_or_token=params.batch_or_token)
    dev_dataset = Dataset(params.src_dev_file, params.src_dev_file,
                          params.src_vocab, params.src_vocab, 1e6,
                          batch_or_token='batch')
    tf.logging.info(
        "End Loading dataset, within {} seconds".format(time.time() - start_time))

    # Build Graph
    with tf.Graph().as_default():
        lr = tf.placeholder(tf.float32, [], "learn_rate")
        train_features = {
            "source": tf.placeholder(tf.int32, [None, None], "source"),
            "target": tf.placeholder(tf.int32, [None, None], "target"),
        }
        eval_features = {
            "source": tf.placeholder(tf.int32, [None, None], "source"),
        }

        # session info
        sess = util.get_session(params.gpus)

        tf.logging.info("Begining Building Training Graph")
        start_time = time.time()

        # create global step
        global_step = tf.train.get_or_create_global_step()

        # set up optimizer
        optimizer = tf.train.AdamOptimizer(lr,
                                           beta1=params.beta1,
                                           beta2=params.beta2,
                                           epsilon=params.epsilon)

        # set up training graph
        loss, gradients = tower_train_graph(train_features, optimizer, params)

        # apply pseudo cyclic parallel operation
        vle, ops = cycle.create_train_op(loss, gradients,
                                         optimizer, global_step, params)

        tf.logging.info("End Building Training Graph, within {} seconds"
                        .format(time.time() - start_time))

        tf.logging.info("Begin Building Inferring Graph")
        start_time = time.time()

        # set up infer graph
        eval_seqs, eval_scores, eval_mask = tower_infer_graph(eval_features, params)

        tf.logging.info("End Building Inferring Graph, within {} seconds"
                        .format(time.time() - start_time))

        # initialize the model
        sess.run(tf.global_variables_initializer())

        # log parameters
        util.variable_printer()

        # create saver
        train_saver = saver.Saver(checkpoints=params.checkpoints,
                                  output_dir=params.output_dir)

        tf.logging.info("Training")
        lrate = params.lrate
        cycle_counter = 1
        cum_loss = []
        cum_gnorm = []

        # restore parameters
        tf.logging.info("Trying restore existing parameters")
        train_saver.restore(sess)

        start_time = time.time()
        for epoch in range(1, params.epoches + 1):

            if epoch < params.recorder.epoch:
                tf.logging.info("Passing {}-th epoch according to record"
                                .format(epoch))
                continue
            params.recorder.epoch = epoch

            tf.logging.info("Training the model for epoch {}".format(epoch))
            size = params.batch_size if params.batch_or_token == 'batch' \
                else params.token_size
            train_batcher = train_dataset.batcher(size,
                                                  buffer_size=params.buffer_size,
                                                  shuffle=params.shuffle_batch)
            train_queue = queuer.EnQueuer(train_batcher)
            train_queue.start(workers=params.nthreads,
                              max_queue_size=params.max_queue_size)

            for lidx, data in enumerate(train_queue.get()):

                if lidx <= params.recorder.lidx:
                    segments = params.recorder.lidx // 5
                    if params.recorder.lidx < 5 or lidx % segments == 0:
                        tf.logging.info("Passing {}-th index according to record"
                                        .format(lidx))
                    continue
                params.recorder.lidx = lidx

                # define feed_dict
                feed_dict = {
                    train_features["source"]: data['src'],
                    train_features["target"]: data['tgt'],
                    lr: lrate,
                }

                if cycle_counter == 1:
                    sess.run(ops["zero_op"])
                if cycle_counter < params.update_cycle:
                    sess.run(ops["collect_op"], feed_dict=feed_dict)
                if cycle_counter == params.update_cycle:
                    cycle_counter = 0
                    _, loss, gnorm, gstep, glr = sess.run(
                        [ops["train_op"], vle["loss"],
                         vle["gradient_norm"], global_step, lr],
                        feed_dict=feed_dict
                    )
                    params.recorder.step = gstep

                    cum_loss.append(loss)
                    cum_gnorm.append(gnorm)

                    if gstep % params.disp_freq == 0:
                        end_time = time.time()
                        tf.logging.info(
                            "{} Epoch {}, GStep {}~{}, LStep {}~{}, "
                            "Loss {:.3f}, GNorm {:.3f}, Lr {:.5f}, Duration {:.3f} s"
                            .format(util.time_str(end_time), epoch,
                                    gstep - params.disp_freq + 1, gstep,
                                    lidx - params.disp_freq * params.update_cycle + 1,
                                    lidx, np.mean(cum_loss), np.mean(cum_gnorm),
                                    glr, end_time - start_time)
                        )
                        start_time = time.time()
                        cum_loss = []
                        cum_gnorm = []

                    if gstep > 0 and gstep % params.eval_freq == 0:
                        eval_start_time = time.time()
                        tranes, scores, indices = evalu.decoding(
                            sess, eval_features, eval_seqs,
                            eval_scores, eval_mask, dev_dataset, params)
                        bleu = evalu.eval_metric(tranes, params.tgt_dev_file,
                                                 indices=indices)
                        eval_end_time = time.time()
                        tf.logging.info(
                            "{} GStep {}, Scores {}, BLEU {}, Duration {:.3f} s"
                            .format(util.time_str(eval_end_time), gstep,
                                    np.mean(scores), bleu,
                                    eval_end_time - eval_start_time)
                        )

                        params.recorder.history_scores.append((gstep,
                                                               float(np.mean(scores))))
                        params.recorder.valid_script_scores.append((gstep,
                                                                    float(bleu)))
                        params.recorder.save_to_json(
                            os.path.join(params.output_dir, "record.json"))

                        # save eval translation
                        evalu.dump_tanslation(
                            tranes,
                            os.path.join(params.output_dir,
                                         "eval-{}.trans.txt".format(gstep)),
                            indices=indices)

                        # handle the learning rate decay in a typical manner
                        history_scores = params.recorder.valid_script_scores
                        history_scores = [score[1] for score in history_scores]
                        # if bleu score stop increasing, half it.
                        if len(history_scores) > 0 and \
                                max(history_scores) > history_scores[-1]:
                            lrate = lrate / 2.

                        train_saver.save(sess, gstep, bleu)

                    if gstep > 0 and gstep % params.sample_freq == 0:
                        decode_seqs, decode_scores, decode_mask = sess.run(
                            [eval_seqs, eval_scores, eval_mask], feed_dict={
                                eval_features["source"]: data['src']
                            })
                        tranes, scores = evalu.decode_hypothesis(decode_seqs, decode_scores,
                                                                 params, mask=decode_mask)
                        for sidx in range(min(5, len(scores))):
                            sample_source = evalu.decode_target_token(
                                data['src'][sidx], params.src_vocab)
                            tf.logging.info("{}-th Source: {}".format(
                                sidx, ' '.join(sample_source)))
                            sample_target = evalu.decode_target_token(
                                data['tgt'][sidx], params.tgt_vocab)
                            tf.logging.info("{}-th Target: {}".format(
                                sidx, ' '.join(sample_target)))
                            sample_trans = tranes[sidx]
                            tf.logging.info("{}-th Translation: {}".format(
                                sidx, ' '.join(sample_trans)))

                    if gstep >= params.max_training_steps:
                        break

                cycle_counter += 1

            train_queue.stop()

            # reset to 0
            params.recorder.lidx = 0

    tf.logging.info("Anyway, your training is finished :)")

    return train_saver.best_score


def evaluate(params):
    # loading dataset
    tf.logging.info("Begin Loading Test Dataset")
    start_time = time.time()
    test_dataset = Dataset(params.src_test_file, params.src_test_file,
                           params.src_vocab, params.src_vocab, 1e6,
                           batch_or_token='batch')
    tf.logging.info(
        "End Loading dataset, within {} seconds".format(time.time() - start_time))

    # Build Graph
    with tf.Graph().as_default():
        eval_features = {
            "source": tf.placeholder(tf.int32, [None, None], "source"),
        }

        # session info
        sess = util.get_session(params.gpus)

        tf.logging.info("Begining Building Evaluation Graph")
        start_time = time.time()

        # set up infer graph
        eval_seqs, eval_scores, eval_mask = tower_infer_graph(eval_features, params)

        tf.logging.info("End Building Inferring Graph, within {} seconds"
                        .format(time.time() - start_time))

        # initialize the model
        sess.run(tf.global_variables_initializer())

        # log parameters
        util.variable_printer()

        # create saver
        eval_saver = saver.Saver(checkpoints=params.checkpoints,
                                 output_dir=params.output_dir)

        # restore parameters
        tf.logging.info("Trying restore existing parameters")
        eval_saver.restore(sess, params.output_dir)

        eval_start_time = time.time()
        tranes, scores, indices = evalu.decoding(
            sess, eval_features, eval_seqs,
            eval_scores, eval_mask, test_dataset, params)
        bleu = evalu.eval_metric(tranes, params.tgt_test_file,
                                 indices=indices)
        eval_end_time = time.time()
        tf.logging.info(
            "{} Scores {}, BLEU {}, Duration {}s"
            .format(util.time_str(eval_end_time),
                    np.mean(scores), bleu, eval_end_time - eval_start_time)
        )

        # save translation
        evalu.dump_tanslation(tranes, params.test_output, indices=indices)

    return bleu
