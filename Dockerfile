FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /workspace

RUN apt update && apt upgrade -y
RUN apt install -y git libgl1-mesa-glx libglib2.0-0
RUN apt-get install -y gcc g++

## dependency installation
COPY ./CLIP ./CLIP
COPY ./requirements.txt ./requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install -r ./requirements.txt

## the following is not needed for build a evaluating image

COPY ./model_output ./model_output
# COPY source code
COPY ./score_utils ./score_utils
COPY ./libs ./libs
COPY ./configs ./configs
COPY ./utils.py ./utils.py

COPY ./sample.py ./sample.py
COPY ./sample.sh ./sample.sh
COPY ./score.py ./score.py
COPY ./Dockerfile ./Dockerfile


## move pretrained weights
# RUN mv ./.insightface ~
# RUN mv .cache/* ~/.cache/