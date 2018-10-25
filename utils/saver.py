# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


class Saver(object):
    def __init__(self,
                 checkpoints=5,   # save the latest number of checkpoints
                 output_dir=None  # the output directory
                 ):
        if output_dir is None:
            output_dir = "./output"
        self.output_dir = output_dir
        self.output_best_dir = os.path.join(output_dir, "best")

        self.saver = tf.train.Saver(
            max_to_keep=checkpoints
        )
        self.best_saver = tf.train.Saver(
            max_to_keep=1
        )
        self.best_score = -1
        self.score_record = tf.gfile.Open(
            os.path.join(self.output_best_dir, "metric.log"),
            mode="a+"
        )

    def save(self, session, step, metric_score):
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MkDir(self.output_dir)
        if not tf.gfile.Exists(self.output_best_dir):
            tf.gfile.MkDir(self.output_best_dir)

        self.saver.save(session,
                        os.path.join(self.output_dir, "model"),
                        global_step=step)

        def _move(path, new_path):
            if tf.gfile.Exists(path):
                if tf.gfile.Exists(new_path):
                    tf.gfile.Remove(new_path)
                tf.gfile.Copy(path, new_path)

        if metric_score > self.best_score:
            self.best_score = metric_score
            self.best_saver.save(
                session, os.path.join(self.output_best_dir, "model"))

            _move(os.path.join(self.output_dir, "param.json"),
                  os.path.join(self.output_best_dir, "param.json"))
            _move(os.path.join(self.output_dir, "record.json"),
                  os.path.join(self.output_best_dir, "record.json"))

            # this recorder only record best scores
            self.score_record.write("Steps {}, Metric Score {}\n"
                                    .format(step, metric_score))

            self.score_record.flush()

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
                            if 'replica' in name:
                                plain_name = name[:name.rfind('/replica')]
                                plain_name = plain_name[len('create_train_op/'):]
                                if reader.has_tensor(plain_name):
                                    tf.logging.info(
                                        '{} get initialization from {}'
                                        .format(name, plain_name))
                                    ops.append(
                                        tf.assign(
                                            var,
                                            reader.get_tensor(plain_name))
                                    )
                                    continue

                            tf.logging.warn("{} is missed".format(name))
                    restore_op = tf.group(*ops, name="restore_global_vars")
                    session.run(restore_op)
