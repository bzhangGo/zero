# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from lrs import lr


class ScoreDecayLr(lr.Lr):
    """Decay the learning rate after each evaluation"""
    def __init__(self,
                 init_lr,
                 min_lr,                # minimum learning rate
                 max_lr,                # maximum learning rate
                 history_scores=None,   # evaluation history metric scores, such as BLEU
                 decay=0.5,             # learning rate decay rate
                 patience=1,            # decay after this number of bad counter
                 name="score_decay_lr"  # model name, no use
                 ):
        super(ScoreDecayLr, self).__init__(init_lr, min_lr, max_lr, name=name)

        self.decay = decay
        self.patience = patience
        self.bad_counter = 0
        self.best_score = -1e9

        if history_scores is not None:
            for score in history_scores:
                self.after_eval(score[1])

    def after_eval(self, eval_score):
        if eval_score > self.best_score:
            self.best_score = eval_score
            self.bad_counter = 0
        else:
            self.bad_counter += 1
            if self.bad_counter >= self.patience:
                self.lrate = self.lrate * self.decay

                self.bad_counter = 0
