#!/usr/bin/env bash

set -e

# First, we need a better solver for conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Create the environment
conda create -y -n dnlp2 python=3.10
conda activate dnlp2

# Install the packages
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm
pip install requests
pip install importlib-metadata
pip install filelock
pip install scikit-learn
pip install tokenizers
pip install explainaboard_client
pip install tensorboard
pip install torch_tb_profiler

# POS and NER tagging
conda install -y -c conda-forge spacy cupy spacy-transformers
pip install spacy-lookups-data
python -m spacy download en_core_web_sm

# For the pre-commit hook
pip install pre-commit

# Make pip happy for triton
pip install cmake
pip install lit

pip install pytorch_optimizer