#!/usr/bin/env bash

# This is an example to show how to finetune your multilingual model with ROBT
# As ROBT requires online decoding, this procedure is very slow; fortunately, ROBT takes few steps to converge.

# set the correct dataset path for the training, either one-to-many path or many-to-many path
data_path=$1
# set the code base directory, indicating the position of source code
zero_path=$2
# set the pretrained model path
pretrain_path=$3
# setup GPU settings
export CUDA_VISIBLE_DEVICES=0

python ${zero_path}/run.py --mode train --parameters=\
hidden_size=512,embed_size=512,filter_size=2048,\
dropout=0.1,label_smooth=0.1,attention_dropout=0.1,relu_dropout=0.1,\
max_len=256,batch_size=80,eval_batch_size=64,\
token_size=5000,batch_or_token='token',\
initializer="uniform_unit_scaling",initializer_gain=1.,\
model_name="transformer_multilingual",scope_name="transformer_multilingual",buffer_size=600000,\
deep_transformer_init=False,enable_robt=True,enable_fuse=True,\
pretrained_model="${pretrain_path}",\
clip_grad_norm=0.0,\
num_heads=8,\
process_num=2,\
lrate=1.0,\
estop_patience=100,\
num_encoder_layer=6,\
num_decoder_layer=6,\
warmup_steps=4000,\
lrate_strategy="noam",\
epoches=5000,\
update_cycle=5,\
gpus=[0],\
disp_freq=1,\
eval_freq=5000,\
sample_freq=1000,\
checkpoints=5,\
max_training_steps=510000,\
beta1=0.9,\
beta2=0.98,\
epsilon=1e-8,\
random_seed=1234,\
src_vocab_file="${data_path}/vocab.zero.src",\
tgt_vocab_file="${data_path}/vocab.zero.tgt",\
to_lang_vocab_file="${data_path}/vocab.zero.lang",\
src_train_file="${data_path}/corpus.train.cmb.bpe.src.shuf",\
tgt_train_file="${data_path}/corpus.train.cmb.bpe.tgt.shuf",\
src_dev_file="${data_path}/corpus.dev.cmb.bpe.src",\
tgt_dev_file="${data_path}/corpus.dev.cmb.bpe.tgt",\
output_dir="train"
