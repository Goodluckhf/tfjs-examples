FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

WORKDIR src/app

RUN apt-get update && \
    apt-get install -y mc curl make screen

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.34.0/install.sh | bash
RUN export NVM_DIR="$HOME/.nvm"