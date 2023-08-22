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
conda install -c conda-forge spacy
conda install -c conda-forge cupy
conda install -c conda-forge spacy-transformers
# packages only available via pip
pip install spacy-lookups-data
python -m spacy download en_core_web_sm

pip install pre-commit

# Make pip happy for triton
pip install cmake
pip install lit