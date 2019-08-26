# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# This is an abstract class that deals with
#   different learning rate decay strategy
# Generally, we decay the learning rate with GPU computation
# However, in this paper, we simply decay the learning rate
#   at CPU level, and feed the decayed lr into GPU for
#   optimization
class Lr(object):
    def __init__(self,
                 init_lrate,        # initial learning rate
                 min_lrate,         # minimum learning rate
                 max_lrate,         # maximum learning rate
                 name="lr",         # learning rate name, no use
                 ):
        self.name = name
        self.init_lrate = init_lrate    # just record the init learning rate
        self.lrate = init_lrate         # active learning rate, change with training
        self.min_lrate = min_lrate
        self.max_lrate = max_lrate

        assert self.max_lrate > self.min_lrate, "Minimum learning rate " \
                                                "should less than maximum learning rate"

    # suppose the eidx starts from 1
    def before_epoch(self, eidx=None):
        pass

    def after_epoch(self, eidx=None):
        pass

    def step(self, step):
        pass

    def after_eval(self, eval_score):
        pass

    def get_lr(self):
        """Return the learning rate whenever you want"""
        return max(min(self.lrate, self.max_lrate), self.min_lrate)
