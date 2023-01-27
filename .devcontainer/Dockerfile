FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN rm -rf /var/lib/apt/lists/*
RUN apt update && apt upgrade -y
RUN apt -y install curl wget vim

##
RUN apt -y install libopencv-dev
RUN apt -y install git

# Anaconda
RUN set -x && \
    wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh && \
    bash Anaconda3-2021.05-Linux-x86_64.sh -b && \
    rm Anaconda3-2021.05-Linux-x86_64.sh

# path setting
ENV PATH $PATH:/root/anaconda3/bin

RUN /root/anaconda3/bin/conda create -y -n godsenv python==3.7
    

WORKDIR /root/workspace
