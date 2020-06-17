#!/usr/bin/env bash

# This is an example to show how to evaluate your multilingual model

# set the correct dataset path for the evaluation, either one-to-many path or many-to-many path
data_path=$1
# set the code base directory, indicating the position of source code
zero_path=$2
# setup GPU settings
export CUDA_VISIBLE_DEVICES=0

# Note for `output_dir="avg"`
# output_dir denotes the trained model's directory; and "avg" denotes averaged checkpoints
# this can be obtained by running ${zero_path}/scripts/checkpoint_averaging.py

# get language information
source ${zero_path}/scripts/data/common.sh

function decode {

src=$1
ref=$2
out=$3
yy=$4

python ${zero_path}/run.py --mode test --parameters=\
hidden_size=512,embed_size=512,filter_size=2048,\
dropout=0.1,label_smooth=0.1,attention_dropout=0.1,relu_dropout=0.1,\
max_len=256,batch_size=80,eval_batch_size=64,\
token_size=5000,batch_or_token='token',\
initializer="uniform_unit_scaling",initializer_gain=1.,\
model_name="transformer_multilingual",scope_name="transformer_multilingual",buffer_size=600000,\
deep_transformer_init=False,enable_robt=False,enable_fuse=True,\
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
max_training_steps=500000,\
beta1=0.9,\
beta2=0.98,\
epsilon=1e-8,\
random_seed=1234,\
src_vocab_file="${data_path}/vocab.zero.src",\
tgt_vocab_file="${data_path}/vocab.zero.tgt",\
to_lang_vocab_file="${data_path}/vocab.zero.lang",\
src_test_file="${src}",\
tgt_test_file="${ref}",\
output_dir="avg",\
test_output="${out}.trans.bpe.txt",\

python3 ${zero_path}/scripts/spm_decode.py \
    --model ${data_path}/sentencepiece.bpe.model \
    --input ${out}.trans.bpe.txt \
    --input_format piece > ${out}.trans.post.txt

scarebleu_options=""
#if [[ ${yy} == "zh" ]]; then
#    scarebleu_options+="--tokenize zh"
#fi

sacrebleu ${ref} ${scarebleu_options} < ${out}.trans.post.txt > ${out}.trans.post.txt.sacrebleu

}


# zero-short evaluation
test_set_name="zero-shot"
for x in ${zero_shot_langs}; do
    for y in ${zero_shot_langs}; do
        if [[ ${x} == ${y} ]]; then
            continue
        fi

        src=${data_path}/test/zero-shot/${x}-${y}/opus.bpe.${x}
        ref=${data_path}/test/zero-shot/${x}-${y}/opus.${y}
        out=${test_set_name}.${x}2${y}.zero

        if [[ ! -f ${src} ]]; then
            continue
        fi

        decode ${src} ${ref} ${out} ${y}
    done
done

# opus, en-xx
src_lang=en
test_set_name="opus"

x=${src_lang}
for y in ${tgt_langs}; do
    if [[ ${x} == ${y} ]]; then
        continue
    fi

    src=${data_path}/test/supervise/${x}-${y}/opus.bpe.${x}
    ref=${data_path}/test/supervise/${x}-${y}/opus.${y}
    out=${test_set_name}.${x}2${y}.zero

    if [[ ! -f ${src} ]]; then
        continue
    fi

    decode ${src} ${ref} ${out} ${y}
done


# reverse generation, xx-en
src_lang=en
test_set_name="opus"

y=${src_lang}
for x in ${tgt_langs}; do
    if [[ ${x} == ${y} ]]; then
        continue
    fi

    src=${data_path}/test/supervise/${x}-${y}/opus.bpe.${x}
    ref=${data_path}/test/supervise/${x}-${y}/opus.${y}
    out=${test_set_name}.${x}2${y}.zero

    if [[ ! -f ${src} ]]; then
        continue
    fi

    decode ${src} ${ref} ${out} ${y}
done
