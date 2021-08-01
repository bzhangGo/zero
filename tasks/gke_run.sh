#! /bin/bash

# example run on gke

PROJECT_ID=translate-research-internship

export DOCKER_IMAGE=gcr.io/${PROJECT_ID}/zero:testing

export CODE_DIR=./

export LOGDIR=gs://biao-bucket/tpu-quickstart/gke/train/log
export BOARDDIR=gs://biao-bucket/tpu-quickstart/gke/train/log/tboards

export TPU_CLUSTER_NAME=tpu-v2-cluster
export TPU_TYPE=v2-32

export GPU_CLUSTER_NAME=p100
export GPU_TYPE=p100

export EXP_NAME=biao.testing.job

export PARAMS="output_dir=gs://biao-bucket/tpu-quickstart/gke/train"

python zero/utils/gke_launch.py \
--base_image=tensorflow:zero_cpu \
--image=$DOCKER_IMAGE \
--logdir=$LOGDIR \
--tboarddir=$BOARDDIR \
--task=train \
--config=wmt14_ende.py \
--parameters=$PARAMS \
--tpu_type=$TPU_TYPE \
--use_tpu \
--runner_cell=$TPU_CLUSTER_NAME \
--name=$EXP_NAME \
--build=$CODE_DIR \
reload all
