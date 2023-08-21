#!/usr/bin/env bash

conda create -n dnlp2 python=3.10
conda activate dnlp2

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install sklearn==0.0
pip install tokenizers==0.13.2
pip install explainaboard_client==0.0.7
pip install tensorboard
conda install -c conda-forge spacy
conda install -c conda-forge cupy
conda install -c conda-forge spacy-transformers
# packages only available via pip
pip install spacy-lookups-data
python -m spacy download en_core_web_sm
