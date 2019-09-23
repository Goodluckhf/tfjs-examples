FROM nvidia/cuda:8.0-cudnn7-runtime-ubuntu16.04

WORKDIR src/app

RUN apt-get update && \
    apt-get install -y mc curl make screen

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.34.0/install.sh | bash
RUN export NVM_DIR="$HOME/.nvm"