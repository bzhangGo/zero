#!/bin/bash

# Download the OPUS-100 dataset from http://data.statmt.org
set -e
set -o pipefail
set -x

path=`pwd`
opus_url=http://data.statmt.org/opus-100-corpus/v1.0/

# # zero codebase path, zero/
zero_path=$1

# define the output directory, allow setting in command line
output_path='OPUS-100'
if [[ "$#" -gt 1 ]]; then
    output_path=$2
fi

# get language information

source ${zero_path}/scripts/data/common.sh

# util functions
function check_make_dir {
    if [[ ! -d $1 ]]; then
        mkdir -p $1
    fi
}

function check_download_file {
    file_url=$1
    file_dst=$2

    if [[ ! -f ${file_dst} ]]; then
        if [[ `wget -S --spider ${file_url}  2>&1 | grep 'HTTP/1.1 200 OK'` ]]; then
            wget ${file_url} -O ${file_dst}
        fi
    fi
}

# download supervised dataset
function handle_supervise {
    x=$1
    y=$2
    data_path=$3
    gen_path=$4

    if [[ `wget -S --spider ${data_path}/$x-$y/opus.$x-$y-train.$x 2>&1 | grep 'HTTP/1.1 200 OK'` ]]; then
        check_make_dir ${gen_path}/$x-$y
    fi
    # training set
    check_download_file ${data_path}/$x-$y/opus.$x-$y-train.$x ${gen_path}/$x-$y/opus.$x-$y-train.$x
    check_download_file ${data_path}/$x-$y/opus.$x-$y-train.$y ${gen_path}/$x-$y/opus.$x-$y-train.$y
    # dev set
    check_download_file ${data_path}/$x-$y/opus.$x-$y-dev.$x ${gen_path}/$x-$y/opus.$x-$y-dev.$x
    check_download_file ${data_path}/$x-$y/opus.$x-$y-dev.$y ${gen_path}/$x-$y/opus.$x-$y-dev.$y
    # test set
    check_download_file ${data_path}/$x-$y/opus.$x-$y-test.$x ${gen_path}/$x-$y/opus.$x-$y-test.$x
    check_download_file ${data_path}/$x-$y/opus.$x-$y-test.$y ${gen_path}/$x-$y/opus.$x-$y-test.$y
}

sup_path=${output_path}/supervised/
check_make_dir ${sup_path}
for lang in ${tgt_langs}; do
    # XX-En
    handle_supervise ${lang} en ${opus_url}/supervised ${sup_path}

    # En-XX
    handle_supervise en ${lang} ${opus_url}/supervised ${sup_path}
done

# download zero-shot dataset
function handle_zero_shot {
    x=$1
    y=$2
    data_path=$3
    gen_path=$4

    if [[ `wget -S --spider ${data_path}/$x-$y/opus.$x-$y-test.$x  2>&1 | grep 'HTTP/1.1 200 OK'` ]]; then
        check_make_dir ${gen_path}/$x-$y
    fi
    # test set
    check_download_file ${data_path}/$x-$y/opus.$x-$y-test.$x ${gen_path}/$x-$y/opus.$x-$y-test.$x
    check_download_file ${data_path}/$x-$y/opus.$x-$y-test.$y ${gen_path}/$x-$y/opus.$x-$y-test.$y
}
zero_path=${output_path}/zero-shot/
check_make_dir ${zero_path}
for z1 in ${zero_shot_langs}; do
    for z2 in ${zero_shot_langs}; do
        if [[ $z1 == $z2 ]]; then
            continue
        fi

        # Z1-Z2
        handle_zero_shot $z1 $z2 ${opus_url}/zero-shot ${zero_path}
    done
done
