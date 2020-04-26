#!/usr/bin/env bash

# extract and prepare the given dataset for multilingual translation
#   including one-to-many (En-XX) and many-to-many (XX-XX) translation

set -e
set -o pipefail
set -x

path=`pwd`

# subword model processing
bpesize=64000

# by default, you have to provide the downloaded dataset path
data_path=$1
# zero codebase path, zero/
zero_path=$2
# whether reuse the sentence piece model for one-to-many and many-to-many
# value: yes/no
bpe_reuse=$3
# setting output directory is permitted
output_path='data'
if [[ "$#" -gt 3 ]]; then
    output_path=$4
fi

bpebin_path=${zero_path}/scripts/

# get language information
source ${zero_path}/scripts/data/common.sh

# util functions
function check_make_dir {
    if [[ ! -d $1 ]]; then
        mkdir -p $1
    fi
}
function add_step {
    step_name=$1
    step_file=$2
    echo ${step_name} >> ${step_file}
}
function check_step {
    step_name=$1
    step_file=$2

    if grep -e "${step_name}" ${step_file}; then
        return 0
    else
        return 1
    fi
}

# set up steps
step1="Step 1: Finish file extraction"
step2="Step 2: Finish bpe training"
step3="Step 3: Finish bpe application"
step4="Step 4: Finish post processing"
step5="Step 5: Finish supervised test set"
step6="Step 6: Finish zero-shot test set"
ckpt_name="checkpoint"

# 1. prepare for one-to-many translation dataset
#    Note the many-to-many data is constructed by concatenating En->XX and XX->En
function handle_supervise {
    x=$1
    y=$2
    in_path=$3
    out_path=$4
    prefix=$5

    # we assume that "x" is English, and "y" is XX, otherwise swap them
    src=$x
    tgt=$y
    if [[ $x != "en" ]]; then
        src=$y
        tgt=$x
    fi

    en_train=${in_path}/$x-$y/opus.$x-$y-train.$src
    xx_train=${in_path}/$x-$y/opus.$x-$y-train.$tgt
    en_dev=${in_path}/$x-$y/opus.$x-$y-dev.$src
    xx_dev=${in_path}/$x-$y/opus.$x-$y-dev.$tgt

    # output (source, target, lang)
    if [[ -f ${en_train} ]]; then
        cat ${en_train} >> ${out_path}/${prefix}.train.en
        cat ${xx_train} >> ${out_path}/${prefix}.train.xx
        cat ${xx_train} | sed "s/^.*$/<2${tgt}>/g" >> ${out_path}/${prefix}.train.lang
    fi
    if [[ -f ${en_dev} ]]; then
        cat ${en_dev} >> ${out_path}/${prefix}.dev.en
        cat ${xx_dev} >> ${out_path}/${prefix}.dev.xx
        cat ${xx_dev} | sed "s/^.*$/<2${tgt}>/g" >> ${out_path}/${prefix}.dev.lang
    fi
}

sup_path=${data_path}/supervised/
one_to_many_path=${output_path}/one-to-many/
check_make_dir ${one_to_many_path}
ckpt_file=${one_to_many_path}/${ckpt_name}
if [[ ! -f ${ckpt_file} ]]; then
    touch ${ckpt_file}
fi

# detection and preparation
if ! check_step "${step1}" ${ckpt_file}; then
    if [[ -f ${one_to_many_path}/corpus.train.lang ]]; then
        rm ${one_to_many_path}/corpus.train.*
        rm ${one_to_many_path}/corpus.dev.*
    fi

    for lang in ${tgt_langs}; do
        # XX-En
        handle_supervise ${lang} en ${sup_path} ${one_to_many_path} corpus

        # En-XX
        handle_supervise en ${lang} ${sup_path} ${one_to_many_path} corpus
    done
    add_step "${step1}" ${ckpt_file}
fi

# after extraction, we get "data.en, data.xx, data.lang"
# learn and apply subword module
if ! check_step "${step2}" ${ckpt_file}; then
    python3 ${bpebin_path}/spm_train.py \
        --input=${one_to_many_path}/corpus.train.en,${one_to_many_path}/corpus.train.xx \
        --model_prefix=${one_to_many_path}/sentencepiece.bpe \
        --vocab_size=${bpesize} \
        --character_coverage=1.0 \
        --model_type=bpe
    add_step "${step2}" ${ckpt_file}
fi

if ! check_step "${step3}" ${ckpt_file}; then
    for dataset in "train" "dev"; do
        python3 ${bpebin_path}/spm_encode.py \
            --model ${one_to_many_path}/sentencepiece.bpe.model \
            --output_format=piece \
            --inputs ${one_to_many_path}/corpus.${dataset}.en ${one_to_many_path}/corpus.${dataset}.xx \
            --outputs ${one_to_many_path}/corpus.${dataset}.bpe.en ${one_to_many_path}/corpus.${dataset}.bpe.xx
    done
    add_step "${step3}" ${ckpt_file}
fi

# post-adjust corpus
if ! check_step "${step4}" ${ckpt_file}; then
    for dataset in  "train" "dev"; do
        paste -d " " ${one_to_many_path}/corpus.${dataset}.lang \
            ${one_to_many_path}/corpus.${dataset}.bpe.en > ${one_to_many_path}/corpus.${dataset}.cmb.bpe.src
        ln -s corpus.${dataset}.bpe.xx ${one_to_many_path}/corpus.${dataset}.cmb.bpe.tgt
    done

    python ${bpebin_path}/shuffle_corpus.py \
        --corpus ${one_to_many_path}/corpus.train.cmb.bpe.src ${one_to_many_path}/corpus.train.cmb.bpe.tgt
    python ${zero_path}/vocab.py \
        --size ${bpesize} ${one_to_many_path}/corpus.train.cmb.bpe.src ${one_to_many_path}/vocab.zero.src
    python ${zero_path}/vocab.py \
        --size ${bpesize} ${one_to_many_path}/corpus.train.cmb.bpe.tgt ${one_to_many_path}/vocab.zero.tgt
    python ${zero_path}/vocab.py \
        --size ${bpesize} ${one_to_many_path}/corpus.train.lang ${one_to_many_path}/vocab.zero.lang

    add_step "${step4}" ${ckpt_file}
fi

# handle test set separately
function handle_supervise_test {
    x=$1
    y=$2
    in_path=$3
    out_path=$4
    model_path=$5
    flag=$6

    # we assume that "x" is English, and "y" is XX, otherwise swap them
    src=$x
    tgt=$y
    if [[ $x != "en" ]]; then
        src=$y
        tgt=$x
    fi

    en_test=${in_path}/$x-$y/opus.$x-$y-test.$src
    xx_test=${in_path}/$x-$y/opus.$x-$y-test.$tgt

    # output (source, target, lang)
    if [[ -f ${en_test} ]]; then
        # src -> tgt
        out_dir=${out_path}/${src}-${tgt}/
        check_make_dir ${out_dir}

        cp ${xx_test} ${out_dir}/opus.${tgt}
        cat ${xx_test} | sed "s/^.*$/<2${tgt}>/g" >> ${out_dir}/opus.lang

        # bpe processing
        python3 ${bpebin_path}/spm_encode.py \
            --model ${model_path}/sentencepiece.bpe.model \
            --output_format=piece \
            --inputs ${en_test} \
            --outputs ${out_dir}/opus.${src}

        paste -d " " ${out_dir}/opus.lang ${out_dir}/opus.${src} > ${out_dir}/opus.bpe.${src}
        rm ${out_dir}/opus.lang ${out_dir}/opus.${src}

        # tgt -> src
        if [[ ${flag} == "reverse" ]]; then
            out_dir=${out_path}/${tgt}-${src}/
            check_make_dir ${out_dir}

            cp ${en_test} ${out_dir}/opus.${src}
            cat ${en_test} | sed "s/^.*$/<2${src}>/g" >> ${out_dir}/opus.lang

            # bpe processing
            python3 ${bpebin_path}/spm_encode.py \
                --model ${model_path}/sentencepiece.bpe.model \
                --output_format=piece \
                --inputs ${xx_test} \
                --outputs ${out_dir}/opus.${tgt}

            paste -d " " ${out_dir}/opus.lang ${out_dir}/opus.${tgt} > ${out_dir}/opus.bpe.${tgt}
            rm ${out_dir}/opus.lang ${out_dir}/opus.${tgt}
        fi
    fi
}

# supervised test set
if ! check_step "${step5}" ${ckpt_file}; then
    for lang in ${tgt_langs}; do
        # XX-En
        handle_supervise_test ${lang} en ${sup_path} ${one_to_many_path}/test/supervise/ ${one_to_many_path} "no"

        # En-XX
        handle_supervise_test en ${lang} ${sup_path} ${one_to_many_path}/test/supervise/ ${one_to_many_path} "no"
    done
    add_step "${step5}" ${ckpt_file}
fi

# handle zero-shot translation evaluation set
function handle_zeroshot {
    x=$1
    y=$2
    in_path=$3
    out_path=$4
    model_path=$5

    src_test=${in_path}/$x-$y/opus.$x-$y-test.$x
    tgt_test=${in_path}/$x-$y/opus.$x-$y-test.$y

    # output (source, target, lang)
    if [[ -f ${src_test} ]]; then
        out_dir=${out_path}/${x}-${y}/
        check_make_dir ${out_dir}

        cp ${tgt_test} ${out_dir}/opus.${y}
        cat ${tgt_test} | sed "s/^.*$/<2${y}>/g" >> ${out_dir}/opus.lang

        # bpe processing
        python3 ${bpebin_path}/spm_encode.py \
            --model ${model_path}/sentencepiece.bpe.model \
            --output_format=piece \
            --inputs ${src_test} \
            --outputs ${out_dir}/opus.${src}

        paste -d " " ${out_dir}/opus.lang ${out_dir}/opus.${src} > ${out_dir}/opus.bpe.${src}
        rm ${out_dir}/opus.lang ${out_dir}/opus.${src}
    fi
}

# for zero-shot test set
if ! check_step "${step6}" ${ckpt_file}; then
    zero_shot_path=${data_path}/zero-shot/
    for z1 in ${zero_shot_langs}; do
        for z2 in ${zero_shot_langs}; do
            if [[ $z1 == $z2 ]]; then
                continue
            fi

            # Z1-Z2
            handle_zeroshot $z1 $z2 ${zero_shot_path} ${one_to_many_path}/test/zero-shot/ ${one_to_many_path}
        done
    done
    add_step "${step6}" ${ckpt_file}
fi


# 2. prepare for many-to-many translation dataset
many_to_many_path=${output_path}/many-to-many/
check_make_dir ${many_to_many_path}
ckpt_file=${many_to_many_path}/${ckpt_name}
if [[ ! -f ${ckpt_file} ]]; then
    touch ${ckpt_file}
fi

if ! check_step "${step1}" ${ckpt_file}; then
    if [[ -f ${many_to_many_path}/corpus.train.tgt ]]; then
        rm ${many_to_many_path}/corpus.train.*
        rm ${many_to_many_path}/corpus.dev.*
    fi

    for dataset in "train" "dev"; do
        cat ${one_to_many_path}/corpus.${dataset}.en >> ${many_to_many_path}/corpus.${dataset}.src
        cat ${one_to_many_path}/corpus.${dataset}.xx >> ${many_to_many_path}/corpus.${dataset}.tgt
        cat ${one_to_many_path}/corpus.${dataset}.xx >> ${many_to_many_path}/corpus.${dataset}.src
        cat ${one_to_many_path}/corpus.${dataset}.en >> ${many_to_many_path}/corpus.${dataset}.tgt
        cat ${one_to_many_path}/corpus.${dataset}.lang >> ${many_to_many_path}/corpus.${dataset}.lang
        cat ${one_to_many_path}/corpus.${dataset}.en | \
            sed "s/^.*$/<2en>/g" >> ${many_to_many_path}/corpus.${dataset}.lang
    done
    add_step "${step1}" ${ckpt_file}
fi

# after extraction, we get "data.src, data.tgt, data.lang"
# learn and apply subword module
if ! check_step "${step2}" ${ckpt_file}; then
    # we learn bpe model only with source data, as source and target share the same data
    if [[ ${bpe_reuse} == "no" ]]; then
        python3 ${bpebin_path}/spm_train.py \
            --input=${many_to_many_path}/corpus.train.src \
            --model_prefix=${many_to_many_path}/sentencepiece.bpe \
            --vocab_size=${bpesize} \
            --character_coverage=1.0 \
            --model_type=bpe
    else
        ln -s $(realpath ${one_to_many_path}/sentencepiece.bpe.*) ${many_to_many_path}/
    fi
    add_step "${step2}" ${ckpt_file}
fi

if ! check_step "${step3}" ${ckpt_file}; then
    for dataset in "train" "dev"; do
        python3 ${bpebin_path}/spm_encode.py \
            --model ${many_to_many_path}/sentencepiece.bpe.model \
            --output_format=piece \
            --inputs ${many_to_many_path}/corpus.${dataset}.src ${many_to_many_path}/corpus.${dataset}.tgt \
            --outputs ${many_to_many_path}/corpus.${dataset}.bpe.src ${many_to_many_path}/corpus.${dataset}.bpe.tgt
    done
    add_step "${step3}" ${ckpt_file}
fi

# post-adjust corpus
if ! check_step "${step4}" ${ckpt_file}; then
    for dataset in  "train" "dev"; do
        paste -d " " ${many_to_many_path}/corpus.${dataset}.lang \
            ${many_to_many_path}/corpus.${dataset}.bpe.src > ${many_to_many_path}/corpus.${dataset}.cmb.bpe.src
        ln -s corpus.${dataset}.bpe.tgt ${many_to_many_path}/corpus.${dataset}.cmb.bpe.tgt
    done

    python ${bpebin_path}/shuffle_corpus.py \
        --corpus ${many_to_many_path}/corpus.train.cmb.bpe.src ${many_to_many_path}/corpus.train.cmb.bpe.tgt
    python ${zero_path}/vocab.py \
        --size ${bpesize} ${many_to_many_path}/corpus.train.cmb.bpe.src ${many_to_many_path}/vocab.zero.src
    python ${zero_path}/vocab.py \
        --size ${bpesize} ${many_to_many_path}/corpus.train.cmb.bpe.tgt ${many_to_many_path}/vocab.zero.tgt
    python ${zero_path}/vocab.py \
        --size ${bpesize} ${many_to_many_path}/corpus.train.lang ${many_to_many_path}/vocab.zero.lang

    add_step "${step4}" ${ckpt_file}
fi

# handle test set separately
if ! check_step "${step5}" ${ckpt_file}; then
    for lang in ${tgt_langs}; do
        # XX-En
        handle_supervise_test ${lang} en ${sup_path} ${many_to_many_path}/test/supervise/ ${many_to_many_path} "reverse"

        # En-XX
        handle_supervise_test en ${lang} ${sup_path} ${many_to_many_path}/test/supervise/ ${many_to_many_path} "reverse"
    done
    add_step "${step5}" ${ckpt_file}
fi

if ! check_step "${step6}" ${ckpt_file}; then
    for z1 in ${zero_shot_langs}; do
        for z2 in ${zero_shot_langs}; do
            if [[ $z1 == $z2 ]]; then
                continue
            fi

            # Z1-Z2
            handle_zeroshot $z1 $z2 ${zero_shot_path} ${many_to_many_path}/test/zero-shot/ ${many_to_many_path}
        done
    done
    add_step "${step6}" ${ckpt_file}
fi

# summarize
echo 'summary:'

echo '1. one-to-many translation'
echo 'path: ' ${one_to_many_path}
echo 'train: ' ${one_to_many_path}/corpus.train.cmb.bpe.src.shuf ${one_to_many_path}/corpus.train.cmb.bpe.tgt.shuf
echo 'dev: ' ${one_to_many_path}/corpus.dev.bpe.src ${one_to_many_path}/corpus.dev.bpe.tgt
echo 'test: ' ${one_to_many_path}/test/supervise ${one_to_many_path}/test/zero-shot

echo '2. many-to-many translation'
echo 'path: ' ${many_to_many_path}
echo 'train: ' ${many_to_many_path}/corpus.train.cmb.bpe.src.shuf ${many_to_many_path}/corpus.train.cmb.bpe.tgt.shuf
echo 'dev: ' ${many_to_many_path}/corpus.dev.bpe.src ${many_to_many_path}/corpus.dev.bpe.tgt
echo 'test: ' ${many_to_many_path}/test/supervise ${many_to_many_path}/test/zero-shot
