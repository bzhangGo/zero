# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


class Saver(object):
    def __init__(self,
                 checkpoints=5,    # save the latest number of checkpoints
                 output_dir=None,  # the output directory
                 best_score=-1,    # the best bleu score before
                 best_checkpoints=1,    # the best checkpoints saved in best checkpoints directory
                 ):
        if output_dir is None:
            output_dir = "./output"
        self.output_dir = output_dir
        self.output_best_dir = os.path.join(output_dir, "best")

        self.saver = tf.train.Saver(
            max_to_keep=checkpoints
        )
        # handle disrupted checkpoints
        if tf.gfile.Exists(self.output_dir):
            ckpt = tf.train.get_checkpoint_state(self.output_dir)
            if ckpt and ckpt.all_model_checkpoint_paths:
                self.saver.recover_last_checkpoints(list(ckpt.all_model_checkpoint_paths))

        self.best_saver = tf.train.Saver(
            max_to_keep=best_checkpoints,
        )
        # handle disrupted checkpoints
        if tf.gfile.Exists(self.output_best_dir):
            ckpt = tf.train.get_checkpoint_state(self.output_best_dir)
            if ckpt and ckpt.all_model_checkpoint_paths:
                self.best_saver.recover_last_checkpoints(list(ckpt.all_model_checkpoint_paths))

        self.best_score = best_score
        # check best bleu result
        metric_dir = os.path.join(self.output_best_dir, "metric.log")
        if tf.gfile.Exists(metric_dir):
            metric_lines = open(metric_dir).readlines()
            if len(metric_lines) > 0:
                best_score_line = metric_lines[-1]
                self.best_score = float(best_score_line.strip().split()[-1])

        # check the top_k_best list and results
        self.topk_scores = []
        topk_dir = os.path.join(self.output_best_dir, "topk_checkpoint")
        ckpt_dir = os.path.join(self.output_best_dir, "checkpoint")
        # direct load the topk information from topk_checkpoints
        if tf.gfile.Exists(topk_dir):
            with tf.gfile.Open(topk_dir) as reader:
                for line in reader:
                    model_name, score = line.strip().split("\t")
                    self.topk_scores.append((model_name, float(score)))
        # backup plan to normal checkpoints and best scores
        elif tf.gfile.Exists(ckpt_dir):
            latest_checkpoint = tf.gfile.Open(ckpt_dir).readline()
            model_name = latest_checkpoint.strip().split(":")[1].strip()
            model_name = model_name[1:-1]  # remove ""
            self.topk_scores.append((model_name, self.best_score))
        self.best_checkpoints = best_checkpoints

        self.score_record = tf.gfile.Open(metric_dir, mode="a+")

    def save(self, session, step, metric_score=None):
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MkDir(self.output_dir)
        if not tf.gfile.Exists(self.output_best_dir):
            tf.gfile.MkDir(self.output_best_dir)

        self.saver.save(session, os.path.join(self.output_dir, "model"), global_step=step)

        def _move(path, new_path):
            if tf.gfile.Exists(path):
                if tf.gfile.Exists(new_path):
                    tf.gfile.Remove(new_path)
                tf.gfile.Copy(path, new_path)

        if metric_score is not None and metric_score > self.best_score:
            self.best_score = metric_score

            _move(os.path.join(self.output_dir, "param.json"),
                  os.path.join(self.output_best_dir, "param.json"))
            _move(os.path.join(self.output_dir, "record.json"),
                  os.path.join(self.output_best_dir, "record.json"))

            # this recorder only record best scores
            self.score_record.write("Steps {}, Metric Score {}\n".format(step, metric_score))
            self.score_record.flush()

        # either no model is saved, or current metric score is better than the minimum one
        if metric_score is not None and \
                (len(self.topk_scores) == 0 or len(self.topk_scores) < self.best_checkpoints or
                 metric_score > min([v[1] for v in self.topk_scores])):
            # manipulate the 'checkpoints', and change the orders
            ckpt_dir = os.path.join(self.output_best_dir, "checkpoint")
            if len(self.topk_scores) > 0:
                sorted_topk_scores = sorted(self.topk_scores, key=lambda x: x[1])
                with tf.gfile.Open(ckpt_dir, mode='w') as writer:
                    best_ckpt = sorted_topk_scores[-1]
                    writer.write("model_checkpoint_path: \"{}\"\n".format(best_ckpt[0]))
                    for model_name, _ in sorted_topk_scores:
                        writer.write("all_model_checkpoint_paths: \"{}\"\n".format(model_name))
                    writer.flush()

                # update best_saver internal checkpoints status
                ckpt = tf.train.get_checkpoint_state(self.output_best_dir)
                if ckpt and ckpt.all_model_checkpoint_paths:
                    self.best_saver.recover_last_checkpoints(list(ckpt.all_model_checkpoint_paths))

            # this change mainly inspired by that sometimes for dataset,
            # the best performance is achieved by averaging top-k checkpoints
            self.best_saver.save(
                session, os.path.join(self.output_best_dir, "model"), global_step=step)

            # handle topk scores
            self.topk_scores.append(("model-{}".format(int(step)), float(metric_score)))
            sorted_topk_scores = sorted(self.topk_scores, key=lambda x: x[1])
            self.topk_scores = sorted_topk_scores[-self.best_checkpoints:]
            topk_dir = os.path.join(self.output_best_dir, "topk_checkpoint")
            with tf.gfile.Open(topk_dir, mode='w') as writer:
                for model_name, score in self.topk_scores:
                    writer.write("{}\t{}\n".format(model_name, score))
                writer.flush()

    def restore(self, session, path=None):
        if path is not None and tf.gfile.Exists(path):
            check_dir = path
        else:
            check_dir = self.output_dir

        checkpoint = os.path.join(check_dir, "checkpoint")
        if not tf.gfile.Exists(checkpoint):
            tf.logging.warn("No Existing Model detected")
        else:
            latest_checkpoint = tf.gfile.Open(checkpoint).readline()
            model_name = latest_checkpoint.strip().split(":")[1].strip()
            model_name = model_name[1:-1]  # remove ""
            model_path = os.path.join(check_dir, model_name)
            model_path = os.path.abspath(model_path)
            if not tf.gfile.Exists(model_path+".meta"):
                tf.logging.error("model '{}' does not exists"
                                 .format(model_path))
            else:
                try:
                    self.saver.restore(session, model_path)
                except tf.errors.NotFoundError:
                    # In this case, we simply assume that the cycle part
                    #   is mismatched, where the replicas are missing.
                    # This would happen if you switch from un-cycle mode
                    #   to cycle mode.
                    tf.logging.warn("Starting Backup Restore")
                    ops = []
                    reader = tf.train.load_checkpoint(model_path)
                    for var in tf.global_variables():
                        name = var.op.name

                        if reader.has_tensor(name):
                            tf.logging.info('{} get initialization from {}'
                                            .format(name, name))
                            ops.append(
                                tf.assign(var, reader.get_tensor(name)))
                        else:
                            tf.logging.warn("{} is missed".format(name))
                    restore_op = tf.group(*ops, name="restore_global_vars")
                    session.run(restore_op)
