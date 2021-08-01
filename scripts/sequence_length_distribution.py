#! /usr/bin/python
# coding: utf-8
# author: Biao Zhang

import sys

"""To illustrate how sequence length distribute in the training data,
to help evaluate TPU batch size
"""


def dist(lens, limit):
  sorted_lens = sorted(lens)
  max_len = min(sorted_lens[-1], limit)

  total_size = len(sorted_lens)

  dix = 0
  for i in range(max_len):
    while dix < total_size and sorted_lens[dix] <= i + 1:
      dix += 1
    print("len {} -> ratio {}".format(i+1, dix*100./total_size))


# input training data
txt_input = sys.argv[1]
# the length upperbound you explore
max_len = int(sys.argv[2])
dist([len(l.strip().split()) for l in open(txt_input, 'r')], max_len)
