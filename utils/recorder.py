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
        record = json.load(open(file_name, 'rb'))
        record = dict((key.encode("UTF-8"), value) for (key, value) in record.items())
        self.__dict__.update(record)

    def save_to_json(self, file_name):
        tf.logging.info("Saving recorder file into {}".format(file_name))
        json.dump(self.__dict__, open(file_name, 'wb'), indent=2)
