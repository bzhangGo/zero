# Run the following commands in order:
#
# ZERO_DIR="path-to-zero/zero"
# ZERO_DEVICE="gpu"  # (Leave empty to build and run CPU only docker) (if use gpu, build docker on a gpu machine)
# docker build --tag tensorflow:zero $(test "$ZERO_DEVICE" = "gpu" && echo "--build-arg base_image=nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04") - < "$ZERO_DIR/docker/dockerfile"
# docker run --rm $(test "$ZERO_DEVICE" = "gpu" && echo "--runtime=nvidia") -it -v ${ZERO_DIR}:path-to-zero/zero -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro -p 6006:6006 -p 8888:8888 --name zero tensorflow:zero bash
#
# backup dependencies to tensorflow 1.13.1

ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Biao Zhang <biaojiaxing@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN apt-get update && apt-get install -y --no-install-recommends \
        aria2 \
        build-essential \
        curl \
        dirmngr \
        git \
        gpg-agent \
        less \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        lsof \
        pkg-config \
        rename \
        rsync \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python 2.7
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776
RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic main" > /etc/apt/sources.list.d/deadsnakes-ppa-bionic.list
RUN apt-get update && apt-get install -y python2.7
RUN update-alternatives --install /usr/bin/python2 python2 /usr/bin/python2.7 1000
# bazel assumes the python executable is "python".
RUN update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1000

RUN curl -O https://bootstrap.pypa.io/get-pip.py && python2 get-pip.py && rm get-pip.py

ARG pip_dependencies=' \
      apache-beam[gcp]>=2.8 \
      contextlib2 \
      google-api-python-client \
      h5py \
      ipykernel \
      jupyter \
      jupyter_http_over_ws \
      matplotlib \
      numpy \
      oauth2client \
      pandas \
      Pillow \
      pyyaml \
      recommonmark \
      scikit-learn==0.20.3 \
      scipy \
      sklearn \
      sphinx \
      sphinx_rtd_theme \
      sympy '

RUN pip2 --no-cache-dir install $pip_dependencies
RUN python2 -m ipykernel.kernelspec

RUN pip2 uninstall -y tensorflow tensorflow-gpu tf-nightly tf-nightly-gpu
RUN pip2 --no-cache-dir install tensorflow==1.13.1

RUN jupyter serverextension enable --py jupyter_http_over_ws

# TensorBoard
EXPOSE 6006

# Jupyter
EXPOSE 8888

WORKDIR "zero"

CMD ["/bin/bash"]
