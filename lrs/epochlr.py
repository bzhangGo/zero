# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from lrs import lr


class EpochDecayLr(lr.Lr):
    """Decay the learning rate after each epoch"""
    def __init__(self,
                 init_lr,
                 min_lr,        # minimum learning rate
                 max_lr,        # maximum learning rate
                 decay=0.5,     # learning rate decay rate
                 name="epoch_decay_lr"
                 ):
        super(EpochDecayLr, self).__init__(init_lr, min_lr, max_lr, name=name)

        self.decay = decay

    def after_epoch(self, eidx=None):
        if eidx is None:
            self.lrate = self.init_lrate * self.decay
        else:
            self.lrate = self.init_lrate * self.decay ** int(eidx)
