#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

data=path-to-preprocessed-ende-dataset/
zero=path-to-zero-codebase/
st_pretrained_model=path-to-st-output-dir/

#the fourth step is to extend the pretrained sentence-level ST to document-level ST

# to improve computation efficiency, here we adopt two steps:
#   1. perform feature extraction using the pretrained ASR Encoder + AFS to largely shorten the audio feature sequence
#   2. perform context-aware ST training using the shortened features

# substep 1. feature extraction
#   this is implemented along with the scorer which outputs both the log scores and AFS-identified features

for dataset in train test dev; do
    # there should be two output files
    # ${dataset}.txt: scoring file
    # ${dataset}.txt.audio.h5: AFS feature file
    python3 ${zero}/run.py --mode score --parameters=\
        max_len=2048,batch_size=80,eval_max_len=2048,eval_batch_size=1,\
        model_name="transformer_afs_st",scope_name="transformer",\
        deep_transformer_init=False,\
        audio_num_mel_bins=40,audio_add_delta_deltas=True,speech_num_feature=120,\
        sinusoid_posenc=False,max_poslen=2048,ctc_enable=False,ctc_alpha=0.3,noise_dropout=0.3,\
        enc_localize="log",dec_localize="none",encdec_localize="none",\
        enable_afs_t=True,enable_afs_f=True,filter_variables=True,\
        src_vocab_file="$data/vocab.zero.en",\
        tgt_vocab_file="$data/vocab.zero.en",\
        src_test_file="$data/${dataset}.audio.h5",\
        tgt_test_file="$data/${dataset}.bpe.en",\
        output_dir="${st_pretrained_model}",\
        test_output="${dataset}.txt"
done

# substep 2. contex-aware ST finetuning
#   crucial parameters:
#   N_src, N_tgt: the number of source and target segmented modeled together
#       Note context length is N_src-1 and N_tgt-1
#   train/test/dev_yaml_file: this is the MuST-C yaml file which contains document information
# set the following, a reasonable setting
N_src=3
N_tgt=3
yaml=path-to-mustc-data/

python3 ${zero}/run.py --mode train --parameters=hidden_size=512,embed_size=512,filter_size=2048,\
dropout=0.5,label_smooth=0.1,attention_dropout=0.2,relu_dropout=0.5,\
max_len=1024,batch_size=80,eval_batch_size=5,\
token_size=6000,batch_or_token='token',\
initializer="uniform_unit_scaling",initializer_gain=1.,\
model_name="transformer_contextual_st",scope_name="transformer",buffer_size=5000,data_leak_ratio=0.1,\
deep_transformer_init=False,\
enable_afs_t=True,enable_afs_f=True,filter_variables=True,\
audio_num_mel_bins=40,audio_add_delta_deltas=True,speech_num_feature=512,\
sinusoid_posenc=False,max_poslen=2048,ctc_enable=False,ctc_alpha=0.3,noise_dropout=0.3,\
enc_localize="log",dec_localize="none",encdec_localize="none",\
N_src=${N_src},N_tgt=${N_tgt},\
asr_pretrain="${st_pretrained_model}",\
clip_grad_norm=0.0,\
l0_norm_warm_up=False,\
l0_norm_context_aware=True,\
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
update_cycle=8,\
gpus=[0],\
disp_freq=1,\
eval_freq=2500,\
save_freq=2500,\
sample_freq=1000,\
checkpoints=5,\
max_training_steps=50000,\
nthreads=8,\
beta1=0.9,\
beta2=0.98,\
swap_memory=True,\
layer_norm=True,\
random_seed=1234,\
src_vocab_file="$data/vocab.zero.en",\
tgt_vocab_file="$data/vocab.zero.de",\
src_train_file="train.txt.audio.h5",\
tgt_train_file="$data/train.bpe.de",\
train_yaml_file="$yaml/train/txt/train.yaml",\
src_dev_file="dev.txt.audio.h5",\
tgt_dev_file="$data/dev.bpe.de",\
dev_yaml_file="$yaml/dev/txt/dev.yaml",\
src_test_file="test.txt.audio.h5",\
tgt_test_file="$data/test.bpe.de",\
test_yaml_file="$yaml/tst-COMMON/txt/tst-COMMON.yaml",\
output_dir="train",\
test_output="",\
