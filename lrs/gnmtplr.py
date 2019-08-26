# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from lrs import lr


class GNMTPDecayLr(lr.Lr):
    """Decay the learning rate during each training step, follows GNMT+"""
    def __init__(self,
                 init_lr,               # initial learning rate
                 min_lr,                # minimum learning rate
                 max_lr,                # maximum learning rate
                 warmup_steps,          # warmup step
                 nstable,               # number of replica
                 lrdecay_start,         # start of learning rate decay
                 lrdecay_end,           # end of learning rate decay
                 name="gnmtp_decay_lr"  # model name, no use
                 ):
        super(GNMTPDecayLr, self).__init__(init_lr, min_lr, max_lr, name=name)

        self.warmup_steps = warmup_steps
        self.nstable = nstable
        self.lrdecay_start = lrdecay_start
        self.lrdecay_end = lrdecay_end

        if nstable < 1:
            raise Exception("Stabled Lrate Value should "
                            "greater than 0, but is {}".format(nstable))

    def step(self, step):
        t = float(step)
        p = float(self.warmup_steps)
        n = float(self.nstable)
        s = float(self.lrdecay_start)
        e = float(self.lrdecay_end)

        decay = np.minimum(1. + t * (n - 1) / (n * p), n)
        decay = np.minimum(decay, n * (2 * n) ** ((s - n * t) / (e - s)))

        self.lrate = self.init_lrate * decay
