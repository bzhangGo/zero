#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

data=path-to-preprocessed-ende-dataset/
zero=path-to-zero-codebase/
afs_pretrained_model=path-to-afs-output-dir/

# the third step is to filter the speech features learned from the AFS encoder
# only feeding those transcript-relevant features for speech translation

# AFS^t: enable_afs_t=True
# AFS^t,f: enable_afs_t=True,enable_afs_f=True
# filter_variables=True: not using ASR decoder parameters, Adam parameters, and embeddings and global_steps
# max_training_steps=30000
# ctc_enable=False: disable CTC training

python3 ${zero}/run.py --mode train --parameters=hidden_size=512,embed_size=512,filter_size=2048,\
dropout=0.2,label_smooth=0.1,attention_dropout=0.1,relu_dropout=0.2,\
max_len=2048,batch_size=80,eval_batch_size=5,\
token_size=10000,batch_or_token='token',\
initializer="uniform_unit_scaling",initializer_gain=1.,\
model_name="transformer_afs_st",scope_name="transformer",buffer_size=5000,data_leak_ratio=0.1,\
deep_transformer_init=False,\
audio_num_mel_bins=40,audio_add_delta_deltas=True,speech_num_feature=120,\
sinusoid_posenc=False,max_poslen=2048,ctc_enable=False,ctc_alpha=0.3,noise_dropout=0.3,\
enc_localize="log",dec_localize="none",encdec_localize="none",\
asr_pretrain="${afs_pretrained_model}/",enable_afs_t=True,enable_afs_f=True,filter_variables=True,\
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
update_cycle=50,\
gpus=[0],\
disp_freq=1,\
eval_freq=2500,\
save_freq=2500,\
sample_freq=1000,\
checkpoints=5,\
max_training_steps=30000,\
nthreads=8,\
beta1=0.9,\
beta2=0.98,\
swap_memory=True,\
layer_norm=True,\
random_seed=1234,\
src_vocab_file="$data/vocab.zero.en",\
tgt_vocab_file="$data/vocab.zero.de",\
src_train_file="$data/train.audio.h5",\
tgt_train_file="$data/train.bpe.de",\
src_dev_file="$data/dev.audio.h5",\
tgt_dev_file="$data/dev.bpe.de",\
src_test_file="$data/test.audio.h5",\
tgt_test_file="$data/test.bpe.de",\
output_dir="train",\
test_output="",\
