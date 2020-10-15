#! /bin/bash

# This example show how to preprocess the audios for MuST-C En-De task


export CUDA_VISIBLE_DEVICES=0

# we extract 40-dimensional log-Mel filterbanks (melbins)

# handle dev set
python t2t_speech.py --melbins 40 --numpad 0 --bs 16 --must_c ./en-de/data/dev dev
# handle test set
python t2t_speech.py --melbins 40 --numpad 0 --bs 16 --must_c ./en-de/data/tst-COMMON test
# handle training set
python t2t_speech.py --melbins 40 --numpad 0 --bs 16 --must_c ./en-de/data/train train
