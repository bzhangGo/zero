# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from lrs import lr


class NoamDecayLr(lr.Lr):
    """Decay the learning rate during each training step, follows Transformer"""
    def __init__(self,
                 init_lr,               # initial learning rate
                 min_lr,                # minimum learning rate
                 max_lr,                # maximum learning rate
                 warmup_steps,          # warmup step
                 hidden_size,           # model hidden size
                 name="noam_decay_lr"   # model name, no use
                 ):
        super(NoamDecayLr, self).__init__(init_lr, min_lr, max_lr, name=name)

        self.warmup_steps = warmup_steps
        self.hidden_size = hidden_size

    def step(self, step):
        step = float(step)
        warmup_steps = float(self.warmup_steps)

        multiplier = float(self.hidden_size) ** -0.5
        decay = multiplier * np.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)
        self.lrate = self.init_lrate * decay
