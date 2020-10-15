# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf


class Recorder(object):
    """To save training processes, inspired by Nematus"""

    def load_from_json(self, file_name):
        tf.logging.info("Loading recoder file from {}".format(file_name))
        with open(file_name, 'r', encoding='utf-8') as fh:
            self.__dict__.update(json.load(fh))

    def save_to_json(self, file_name):
        tf.logging.info("Saving recorder file into {}".format(file_name))
        with open(file_name, 'w', encoding='utf-8') as fh:
            json.dump(self.__dict__, fh, indent=2)
