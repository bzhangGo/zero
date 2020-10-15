#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

data=path-to-preprocessed-ende-dataset/
zero=path-to-zero-codebase/
asr_pretrained_model=path-to-asr-output-dir/

# the second step is to filter the speech features learned from the AFS encoder
# only feeding those transcript-relevant features for speech translation

# crucial hyperparameters in this step:
# asr_pretrain="${asr_pretrained_model}/": setting the pretrained ASR model path, this is the `output_dir` of ASR pretraining
# The following steps are for L0Drop, pruning with \lambda=0.5
# l0_norm_warm_up=False,
# l0_norm_reg_scalar=0.5,

# AFS^t: enable_afs_t=True
# AFS^t,f: enable_afs_t=True,enable_afs_f=True
# filter_variables=False: allow full initialization for all encoder-decoder parameters
# max_training_steps=35000: finetuning 5000 extra steps
# ctc_enable=False: disable CTC training

python3 ${zero}/run.py --mode train --parameters=hidden_size=512,embed_size=512,filter_size=2048,\
dropout=0.2,label_smooth=0.1,attention_dropout=0.1,relu_dropout=0.2,\
max_len=2048,batch_size=80,eval_batch_size=5,\
token_size=14000,batch_or_token='token',\
initializer="uniform_unit_scaling",initializer_gain=1.,\
model_name="transformer_afs",scope_name="transformer",buffer_size=5000,data_leak_ratio=0.1,\
deep_transformer_init=False,\
audio_num_mel_bins=40,audio_add_delta_deltas=True,speech_num_feature=120,\
sinusoid_posenc=False,max_poslen=2048,ctc_enable=False,ctc_alpha=0.3,noise_dropout=0.3,\
enc_localize="log",dec_localize="none",encdec_localize="none",\
asr_pretrain="${asr_pretrained_model}/",enable_afs_t=True,enable_afs_f=True,filter_variables=False,\
clip_grad_norm=0.0,\
l0_norm_warm_up=False,\
l0_norm_reg_scalar=0.5,\
num_heads=8,\
process_num=4,\
lrate=1.0,\
estop_patience=100,\
num_encoder_layer=6,\
num_decoder_layer=6,\
warmup_steps=4000,\
lrate_strategy="noam",\
epoches=5000,\
update_cycle=36,\
gpus=[0],\
disp_freq=1,\
eval_freq=2500,\
save_freq=2500,\
sample_freq=1000,\
checkpoints=5,\
max_training_steps=35000,\
nthreads=8,\
beta1=0.9,\
beta2=0.98,\
swap_memory=True,\
layer_norm=True,\
random_seed=1234,\
src_vocab_file="$data/vocab.zero.en",\
tgt_vocab_file="$data/vocab.zero.en",\
src_train_file="$data/train.audio.h5",\
tgt_train_file="$data/train.bpe.en",\
src_dev_file="$data/dev.audio.h5",\
tgt_dev_file="$data/dev.bpe.en",\
src_test_file="$data/test.audio.h5",\
tgt_test_file="$data/test.bpe.en",\
output_dir="train",\
test_output="",\
