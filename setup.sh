#!/usr/bin/env bash

set -e

conda create -y -n dnlp2 python=3.10
conda activate dnlp2

conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm
pip install requests
pip install importlib-metadata
pip install filelock
pip install scikit-learn
pip install tokenizers
pip install explainaboard_client
pip install tensorboard
pip install pre-commit

# Make pip happy for triton
pip install cmake
pip install lit