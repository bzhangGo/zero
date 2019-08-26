# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from lrs import lr


class VanillaLR(lr.Lr):
    """Very basic learning rate, constant learning rate"""
    def __init__(self,
                 init_lr,       # learning rate
                 min_lr,        # minimum learning rate
                 max_lr,        # maximum learning rate
                 name="vanilla_lr"
                 ):
        super(VanillaLR, self).__init__(init_lr, min_lr, max_lr, name=name)
