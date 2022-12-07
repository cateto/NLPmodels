FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

ARG model_type=kobert
ARG model_name="monologg/kobert"
ARG corpus_path="/repository/NER/data/k_corpus/corpus_v3.json"
ARG export_path="/repository/NER/outputs/"

RUN apt-get update && apt-get upgrade -y && apt-get clean

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 && \
    update-alternatives --set python /usr/bin/python3.7 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    update-alternatives --set python3 /usr/bin/python3.7

RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

RUN apt-get install -y python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY python /lab_ner/python

WORKDIR /lab_ner/python
RUN pip install --default-timeout=300 --no-cache-dir -e .

WORKDIR /lab_ner/python/src

RUN python python run_train.py --model_type ${model_type} --model_name ${model_name} --corpus_path ${corpus_path} --export_path ${export_path}


