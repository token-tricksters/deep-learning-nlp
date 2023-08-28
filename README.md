# DNLP SS23 Final Project - Multitask BERT - Token Tricksters

This repository is the official implementation of the Multitask BERT project for the Deep Learning for Natural Language
Processing course at the University of Göttingen by Lars Kaesberg, Niklas Bauer, Constantin Dalinghaus, and Sebastian
Kampen.

> 📋 Optional: include a graphic explaining your approach/main result

## Requirements

To install requirements, using conda, run:

```sh
source setup.sh
```

The script will create a new conda environment called `dnlp2` and install all required packages. The environment is
activated with `conda activate dnlp2`.
We use Python 3.10 and PyTorch 2.0+.

## Training

To train the model, activate the environment and run this command:

```sh
python -u multitask_classifier.py --use_gpu --lr 1e-3 --batch_size 64
```

There are a lot of parameters that can be set. To see all of them, run `python multitask_classifier.py --help`. The most
important ones are:

| Parameter             | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| `--use_gpu`           | Whether to use the GPU.                                        |
| `--lr`                | Learning rate.                                                 |
| `--batch_size`        | Batch size.                                                    |
| `--epochs`            | Number of epochs.                                              |
| `--optimizer`         | Optimizer to use. Options are `AdamW` and `SophiaW`.           |
| `--scheduler`         | Learning rate scheduler to use.                                |
| `--unfreeze_interval` | Number of epochs until the next BERT layer is unfrozen         |
| `--additional_input`  | Activates the usage for POS and NER tags for the input of BERT |

> 📋 Describe how to train the models, with example commands on how to train the models in your paper, including the full
> training procedure and appropriate hyperparameters.

## Evaluation

The model is evaluated after each epoch on the validation set. The results are printed to the console and saved in
the `logdir` directory. The best model is saved in the `models` directory.

## Methodology

This section describes the methodology used in our experiments to extend the training of the multitask BERT model to the three tasks of paraphrase identification, sentiment classification, and semantic textual similarity.

A pretrained BERT ([BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)) model was used as the basis for our experiments. The model was fine-tuned on the three tasks using a multitask learning approach. The model was trained on the three tasks simultaneously, with a single shared BERT encoder and three separate task-specific classifiers.

### POS and NER Tag Embeddings

Based on [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606), which showed that the addition of subword information to word embeddings can improve performance on downstream tasks, we extended our approach by incorporating Part-of-Speech (POS) and Named Entity Recognition (NER) tag embeddings into the input representation. The primary goal was to investigate whether the inclusion of linguistic information could lead to improved performance on the tasks.

For each input sentence we used a POS tagger and a NER tagger to generate corresponding tags. These tags were then converted into embeddings and added to the existing word embeddings before being fed into the BERT model. During training, the POS and NER tags were dynamically generated for each batch of data. To increase training efficiency, we implemented a caching mechanism where the computed tag embeddings were stored and reused across multiple epochs.

#### Implementation

For the efficient and accurate tagging of POS and NER, we used the [spaCy](https://spacy.io/) library. spaCy is a popular NLP library that offers pre-trained models for various NLP tasks, including POS and NER tagging.

The tagging process occurs during data preprocessing, where each sentence is tokenized into individual words. The spaCy pipeline is then invoked to annotate each word with its corresponding POS tag and NER label. The resulting tags and labels are subsequently converted into embeddings that are seamlessly integrated into the existing word embeddings.

#### Experimental Results

Contrary to our initial expectations, the inclusion of POS and NER tag embeddings did not yield the desired improvements across the three tasks. Experimental results indicated that the performance either remained stagnant or even slightly deteriorated compared to the baseline BERT model without tag embeddings.

#### Impact on Training Process

An additional observation was the notable increase in training time when incorporating POS and NER tag embeddings. This extended training time was attributed to the additional computational overhead required for generating and embedding the tags.

#### Conclusion

Although the integration of POS and NER tag embeddings initially seemed promising, our experiments showed that this approach did not contribute to performance across tasks. The training process was noticeably slowed down by the inclusion of tag embeddings.

As a result, we concluded that the benefits of incorporating POS and NER tags were not substantial enough to justify the extended training time. Future research could explore alternative ways of effectively exploiting linguistic features while minimising the associated computational overhead.

One possible explanation for the lack of performance improvements could be that the BERT model already encodes some syntactic information in its word embeddings. [A Structural Probe for Finding Syntax in Word Representations](https://aclanthology.org/N19-1419.pdf) showed that some syntactic information is already encoded in the word embeddings of pretrained BERT models, which could explain why the inclusion of POS and NER tags did not lead to performance improvements.

### Sophia

We implemented the Sophia (**S**econd-**o**rder Cli**p**ped Stoc**h**astic Opt**i**miz**a**tion) optimizer completly from scratch, which is a second-order optimizer for language model pre-training. The paper promises convergence twice as fast as AdamW and better generalisation performance. It uses a light weight estimate of the diagonal of the Hessian matrix to approximate the curvature of the loss function. It also uses clipping to control the worst-case update size. By only updating the Hessian estimate every few iterations, the overhead is negligible.

The optimizer was introduced recently in the paper [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342).

#### Implementation

The paper describes the optimizer in detail, but does not provide any usable code. We implemented the optimizer from scratch in PyTorch. The optimizer is implemented in the [`optimizer.py`](optimizer.py) file and can be used in the multitask classifier by setting the `--optimizer` parameter.

There are two ways of estimating the Hessian. The first option is to use the Gauss-Newton-Bartlett approximation, which is computed using an average over the minibatch gradients. However, this estimator requires the existence of a multi-class classification problem from which to sample. This is not the case for some of our tasks, e.g. STS, which is a regression task. The estimator is still implemented as `SophiaG`.

The second option is to use Hutchinson's unbiased estimator of the Hessian diagonal by sampling from a spherical Gaussian distribution. This estimator is implemented as `SophiaH`. This estimator can be used for all tasks. It requires a Hessian vector product, which is implemented in most modern deep learning frameworks, including PyTorch.

#### Convergence

While the implementation of this novel optimizer was a challenge, the results were not as promised. The optimizer did not converge faster than AdamW, and the performance was comparable. This could be due to the fact that the optimizer was designed for pre-training language models, which is a different task to ours.

A more recent paper studing different training algorithms for transformer-based language models ([No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models](https://arxiv.org/pdf/2307.06440.pdf)) comes to the conclusion that the training algorithm gains vanish with a fully decayed learning rate. They show performance being about the same as the baseline (AdamW), which is what we observed.

### Synthetic Data

## Experiments

### Classifier

In this section, we take a look at the classifiers and the attention layer that are important components of our multi-task learning architecture. These elements work in synergy to process the output from the BERT model and make predictions for the specific tasks of Sentiment Analysis, Paraphrase Detection, and Semantic Textual Similarity.

#### Attention Layer

The attention mechanism plays a major role in capturing and emphasizing salient information within the output embeddings generated by the BERT model. We implemented an `AttentionLayer` that accepts the last hidden state of the BERT output and applies a weighted sum mechanism to enhance the importance of certain tokens while suppressing others. This layer aids in creating a more focused representation of the input sentence, which is crucial for effective downstream tasks.

#### Sentiment Analysis Classifier

The Sentiment Analysis classifier aims to predict the sentiment polarity of a given sentence. The classifier architecture consists of multiple linear layers that refine the BERT embeddings. After passing through these layers, the embeddings are projected into a space where they can be classified into five sentiment classes: negative, somewhat negative, neutral, somewhat positive, and positive. The final linear layer produces logits corresponding to each sentiment class, which are then used to compute the predicted sentiment label.

#### Paraphrase Detection Classifier

The Paraphrase Detection classifier focuses on determining whether two input sentences are paraphrases of each other. To achieve this, we employ a two-step process. First, the BERT embeddings for each input sentence are separately processed through a linear layer. Subsequently, we compute the absolute difference and absolute sum of these processed embeddings. These two concatenated features are then fed through additional linear layers to generate logits for predicting paraphrases.

#### Semantic Textual Similarity Estimator

For the Semantic Textual Similarity (STS) task, our approach relies on cosine similarity to gauge the degree of semantic similarity between two input sentences. The BERT embeddings for the input sentences are generated and then compared using cosine similarity. The resulting similarity score is scaled to range between 0 and 5, providing an estimate of how semantically similar the two sentences are.

#### Conclusion

The attention layer and the individual classifiers for Sentiment Analysis, Paraphrase Detection, and Semantic Textual Similarity work together harmoniously to extract relevant features from the BERT embeddings and make predictions for the respective tasks. Each classifier's architecture is tailored to the unique characteristics of its task, enabling our multi-task learning framework to address multiple NLP challenges simultaneously.

By leveraging attention mechanisms and task-specific classifiers, our architecture strives to capitalize on the power of BERT embeddings and advance the field of NLP by achieving competitive performance across Sentiment Analysis, Paraphrase Detection, and Semantic Textual Similarity tasks.

### Layer Unfreeze

### Mixture of Experts

### Automatic Mixed Precision

## Results

Our multitask model achieves the following performance on:

### [Paraphrase Identification on Quora Question Pairs](https://paperswithcode.com/sota/paraphrase-identification-on-quora-question)

Paraphrase Detection is the task of finding paraphrases of texts in a large corpus of passages.
Paraphrases are “rewordings of something written or spoken by someone else”; paraphrase
detection thus essentially seeks to determine whether particular words or phrases convey
the same semantic meaning.

| Model name       | Parameters   | Accuracy |
|------------------|--------------|----------|
| My awesome model | SophiaW 1e-3 | 99.99%   |

### [Sentiment Classification on Stanford Sentiment Treebank (SST)](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained)

A basic task in understanding a given text is classifying its polarity (i.e., whether the expressed
opinion in a text is positive, negative, or neutral). Sentiment analysis can be utilized to
determine individual feelings towards particular products, politicians, or within news reports.
Each phrase has a label of negative, somewhat negative,
neutral, somewhat positive, or positive.

| Model name       | Parameters   | Accuracy |
|------------------|--------------|----------|
| My awesome model | SophiaW 1e-3 | 99.99%   |

### [Semantic Textual Similarity on STS](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark)

The semantic textual similarity (STS) task seeks to capture the notion that some texts are
more similar than others; STS seeks to measure the degree of semantic equivalence [Agirre
et al., 2013]. STS differs from paraphrasing in it is not a yes or no decision; rather STS
allows for 5 degrees of similarity.

| Model name       | Parameters   | Pearson Correlation |
|------------------|--------------|---------------------|
| My awesome model | SophiaW 1e-3 | 0.9                 |

> 📋 Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main
> result is a figure, include that figure and link to the command or notebook to reproduce it.

## Contributors

| Lars Kaesberg | Niklas Bauer | Constantin Dalinghaus | Sebastian Kampen |
|---------------|--------------|-----------------------|------------------|
|               |              |                       |                  |

## Contributing

The project involves the creation of software and documentation to be released under an open source licence.
This license is the Apache License 2.0, which is a permissive licence that allows the use of the software for
commercial purposes. The licence is also compatible with the licences of the libraries used in the project.

To contribute to the project, please follow the following steps:

Clone the repository to your local machine.

````sh
git clone git@gitlab.gwdg.de:deep-learning-nlp/token-tricksters.git
````

Add the upstream repository as a remote and disable pushing to it. This allows you to pull from the upstream repository
but not push to it.

````sh
git remote add upstream https://github.com/truas/minbert-default-final-project
git remote set-url --push upstream DISABLE
````

If you want to pull from the upstream repository you can use the following commands.

````sh
git fetch upstream
git merge upstream/main
````

### Pre-Commit Hooks

The code quality is checked with pre-commit hooks. To install the pre-commit hooks run the following command.
This is used to ensure that the code quality is consistent and that the code is formatted uniformly.

````sh
pip install pre-commit
pre-commit install
````

This will install the pre-commit hooks in your local repository. The pre-commit hooks will run automatically before each
commit. If the hooks fail the commit will be aborted. You can skip the pre-commit hooks by adding the `--no-verify` flag
to your commit command.

The installed pre-commit hooks are:

- [`black`](https://github.com/psf/black) - Code formatter (Line length 100)
- [`flake8`](https://github.com/PyCQA/flake8) Code linter (Selected rules)
- [`isort`](https://github.com/PyCQA/isort) - Import sorter

### Grete Cluster

To run the multitask classifier on the Grete cluster you can use the `run_train.sh` script. You can change the
parameters in the script to your liking. To submit the script use

````sh
sbatch run_train.sh
````

To check on your job you can use the following command

```sh
squeue --me
```

The logs of your job will be saved in the `logdir` directory. The best model will be saved in the `models` directory.

To run tensorboard on the Grete cluster you can use the following commands to create a tunnel to your local machine and
start tensorboard.

````sh
ssh -L localhost:16006:localhost:6006 bzkurs42@glogin9.hlrn.de
module load anaconda3
source activate dnlp2
tensorboard --logdir logdir
````

If you want to run the model on the Grete cluster interactively you can use the following command, which will give you
access to a GPU node with an A100 GPU. This is for testing purposes only and should not be used for training.

````sh
srun -p grete:shared --pty -G A100:1 --interactive bash
````

## Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the
Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun,
John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon
University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)
library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/)
and [Terry Ruas](https://terryruas.com/).
