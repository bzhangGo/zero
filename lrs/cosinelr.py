# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from lrs import lr


class CosineDecayLr(lr.Lr):
    """Decay the learning rate during each training step, follows FairSeq"""
    def __init__(self,
                 init_lr,               # initial learning rate => warmup_init_lr
                 min_lr,                # minimum learning rate
                 max_lr,                # maximum learning rate
                 warmup_steps,          # warmup step   => warmup_updates
                 decay,                 # learning rate shrink factor for annealing
                 t_mult=1,              # factor to grow the length of each period
                 update_period=5000,    # initial number of updates per period
                 name="cosine_decay_lr"  # model name, no use
                 ):
        super(CosineDecayLr, self).__init__(init_lr, min_lr, max_lr, name=name)

        self.warmup_steps = warmup_steps

        self.warmup_init_lr = init_lr
        self.warmup_end_lr = max_lr
        self.t_mult = t_mult
        self.period = update_period

        if self.warmup_steps > 0:
            self.lr_step = (self.warmup_end_lr - self.warmup_init_lr) / self.warmup_steps
        else:
            self.lr_step = 1.

        self.decay = decay

        # initial learning rate
        self.lrate = init_lr

    def step(self, step):
        if step < self.warmup_steps:
            self.lrate = self.warmup_init_lr + step * self.lr_step
        else:
            curr_updates = step - self.warmup_steps
            if self.t_mult != 1:
                i = math.floor(math.log(1 - curr_updates / self.period * (1 - self.t_mult), self.t_mult))
                t_i = self.t_mult ** i * self.period
                t_curr = curr_updates - (1 - self.t_mult ** i) / (1 - self.t_mult) * self.period
            else:
                i = math.floor(curr_updates / self.period)
                t_i = self.period
                t_curr = curr_updates - (self.period * i)

            lr_shrink = self.decay ** i
            min_lr = self.min_lrate * lr_shrink
            max_lr = self.max_lrate * lr_shrink

            self.lrate = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t_curr / t_i))

        return self.lrate
