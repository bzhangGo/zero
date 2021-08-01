# coding: utf-8
# author: Biao Zhang

import sys

meta_path = sys.argv[1]   # o2m/m2o meta information path
trans_dir = sys.argv[2]   # en2xx or xx2en
hypothesis = sys.argv[3]  # model's translation

outputs = open(hypothesis, 'r').readlines()
references = open(meta_path + "/%s.ref" % trans_dir, 'r').readlines()

cnts = open(meta_path + "/%s.cnt" % trans_dir, 'r').readlines()
cnts = [int(c.strip().split()[0]) for c in cnts]

names = open(meta_path + "/%s.name" % trans_dir, 'r').readlines()


def dump(data, f):
  with open(f, 'w') as writer:
    for d in data:
      writer.write(d.strip() + "\n")


start_idx = 0
for name, cnt in zip(names, cnts):
  name = name.strip()

  end_idx = start_idx + cnt

  dec_data = outputs[start_idx:end_idx]
  ref_data = references[start_idx:end_idx]

  start_idx = end_idx

  dump(dec_data, '%s.trans.txt' % name)
  dump(ref_data, '%s.ref.txt' % name)

assert start_idx == len(outputs)
