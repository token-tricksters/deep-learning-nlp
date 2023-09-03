#!/usr/bin/bash -l

set -e

# First, we need a better solver for conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Create the environment
conda create -y -n dnlp2 python=3.10
conda activate dnlp2

# Make pip happy for triton
pip install cmake lit

# Install the packages
conda install -y pytorch=2.0 torchvision=0.15 torchaudio=2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install "tqdm>=4.66" requests importlib-metadata "filelock>=3.12" "scikit-learn>=1.3" "tokenizers>=0.13" "explainaboard_client>=0.1" "tensorboard>=2.14" "torch_tb_profiler>=0.4" "pytorch_optimizer>=2.11"

# POS and NER tagging
conda install -y -c conda-forge "spacy>=3.5" cupy spacy-transformers
pip install "spacy-lookups-data>=1.0"
python -m spacy download en_core_web_sm

# For the pre-commit hook
pip install pre-commit

# For the hyperparameter tuning
pip install "ray[tune]>=2.6" "optuna>=3.3"
