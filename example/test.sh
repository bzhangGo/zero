#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

data=path-to-preprocessed-ende-dataset/
zero=path-to-zero-codebase/
moses=path-to-mosesdecoder/
yaml=path-to-mustc-data/

# average last 5 checkpoints
python3 ${zero}/scripts/checkpoint_averaging.py --path ../train --output avg --checkpoints 5 --gpu 0

# to perform decoding for context-aware ST models
#   remember to control the following hyperparamters for different inference mode
#   N_src,N_tgt: the context used for training
#   inference_mode:
#       cbd: chunk-based decoding
#       swbd: sliding-window based decoding
#       swbd_cons and sent_prob=0.0:    sliding-window based decoding with prefix constraint
#       imed: in-model ensemble decoding, we often set sent_prob=0.5
N_src=3
N_tgt=3
# note imed and swbd_cons is slow, turn to cbd or swbd for fast test
inference_mode="imed"
sent_prob=0.5

python3 ${zero}/run.py --mode test --parameters=hidden_size=512,embed_size=512,filter_size=2048,\
dropout=0.2,label_smooth=0.1,attention_dropout=0.1,relu_dropout=0.2,\
max_len=2048,batch_size=80,eval_batch_size=5,\
token_size=15000,batch_or_token='token',\
initializer="uniform_unit_scaling",initializer_gain=1.,\
model_name="transformer_contextual_st",scope_name="transformer",buffer_size=5000,data_leak_ratio=0.1,\
deep_transformer_init=False,\
clip_grad_norm=0.0,\
N_src=${N_src},N_tgt=${N_tgt},inference_mode=${inference_mode},sent_prob=${sent_prob},\
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
max_training_steps=30000,\
nthreads=8,\
beta1=0.9,\
beta2=0.98,\
swap_memory=True,\
layer_norm=True,\
random_seed=1234,\
src_vocab_file="$data/vocab.zero.en",\
tgt_vocab_file="$data/vocab.zero.de",\
src_test_file="$data/test.audio.h5",\
tgt_test_file="$data/test.bpe.de",\
test_yaml_file="$yaml/tst-COMMON/txt/tst-COMMON.yaml",\
output_dir="avg",\
test_output="trans.txt",\

# post processing
sed -r 's/ \@(\S*?)\@ /\1/g' < trans.txt |
sed -r 's/\@\@ //g' |
sed "s/&lt;s&gt;//" |
${moses}/scripts/recaser/detruecase.perl > trans.tok.txt

# evaluation
${moses}/scripts/generic/multi-bleu.perl $data/test.reftok.de < trans.tok.txt > test.bleu
