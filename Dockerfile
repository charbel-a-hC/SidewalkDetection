FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

# To save you a headache
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Fix Nvidia/Cuda repository key rotation
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/*
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*  
RUN apt-key del 7fa2af80 &&\
	apt-get update && \
	apt-get  install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb 

RUN apt update && apt install -y software-properties-common

# Install System Dependencies
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update

RUN apt-get install -y python3.7 \
    python3-pip \
    python3.7-venv \
    python3.7-dev \
    python3.7-distutils \
    curl \
    vim \
    git

# Adjust default python3 version to required version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
# Update pip3 version
RUN python3 -m pip install --upgrade pip

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=/root/.local/bin:$PATH
RUN poetry config virtualenvs.create false


WORKDIR /SidewalkDetection
#COPY pyproject.toml .
#COPY Makefile .
#RUN make env-docker

