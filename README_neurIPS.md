# DNLP SS23 Final Project - Multitask BERT - Token Tricksters

This repository is the official implementation of the Multitask BERT - Token Tricksters project for the Deep Learning for Natural Language Processing course at the University of Göttingen.

> 📋 Optional: include a graphic explaining your approach/main result

## Requirements

To install requirements:

```sh
./setup.sh
```

The script will create a new conda environment called `dnlp2` and install all required packages. The environment can be activated with `conda activate dnlp2`.
We use Python 3.10 and PyTorch 2.0+.

## Training

To train the model, activate the environment and run this command:

```sh
python -u multitask_classifier.py --use_gpu --lr 1e-3 --batch_size 64
```

There are a lot of parameters that can be set. To see all of them, run `python multitask_classifier.py --help`. The most important ones are:

| Parameter | Description |
|-----------|-------------|
| `--use_gpu` | Whether to use the GPU. |
| `--lr` | Learning rate. |
| `--batch_size` | Batch size. |
| `--epochs` | Number of epochs. |
| `--optimizer` | Optimizer to use. Options are `AdamW` and `SophiaW`. |
| `--scheduler` | Learning rate scheduler to use. |

> 📋 Describe how to train the models, with example commands on how to train the models in your paper, including the full
> training procedure and appropriate hyperparameters.

## Evaluation

The model is evaluated after each epoch on the validation set. The results are printed to the console and saved in the `logdir` directory. The best model is saved in the `models` directory.

## Methodology

## Experiments

## Results

Our multitask model achieves the following performance on :

### [Paraphrase Identification on Quora Question Pairs](https://paperswithcode.com/sota/paraphrase-identification-on-quora-question)
Paraphrase Detection is the task of finding paraphrases of texts in a large corpus of passages.
Paraphrases are “rewordings of something written or spoken by someone else”; paraphrase
detection thus essentially seeks to determine whether particular words or phrases convey
the same semantic meaning.

| Model name       | Parameters     | Accuracy       |
|------------------|----------------|----------------|
| My awesome model | SophiaW 1e-3       | 99.99%         |

### [Sentiment Classification on Stanford Sentiment Treebank (SST)](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained)
A basic task in understanding a given text is classifying its polarity (i.e., whether the expressed
opinion in a text is positive, negative, or neutral). Sentiment analysis can be utilized to
determine individual feelings towards particular products, politicians, or within news reports.
Each phrase has a label of negative, somewhat negative,
neutral, somewhat positive, or positive.

| Model name       | Parameters     | Accuracy       |
|------------------|----------------|----------------|
| My awesome model | SophiaW 1e-3       | 99.99%         |

### [Semantic Textual Similarity on STS](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark)
The semantic textual similarity (STS) task seeks to capture the notion that some texts are
more similar than others; STS seeks to measure the degree of semantic equivalence [Agirre
et al., 2013]. STS differs from paraphrasing in it is not a yes or no decision; rather STS
allows for 5 degrees of similarity.

| Model name       | Parameters     | Pearson Correlation       |
|------------------|----------------|----------------|
| My awesome model | SophiaW 1e-3       | 0.9          |

> 📋 Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main
> result is a figure, include that figure and link to the command or notebook to reproduce it.

## Contributors

| Lars Kaesberg | Niklas Bauer | Constantin Dalinghaus | Sebastian Kampen |
|---------------|--------------|-----------------------|------------------|
|               |              |                       |                  |

## Contributing

> 📋 Pick a licence and describe how to contribute to your code repository.

## Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!) 

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).
