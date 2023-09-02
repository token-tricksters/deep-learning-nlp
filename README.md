# DNLP SS23 Final Project - Multitask BERT
    
<div align="right">
<u> Token Tricksters </u> <br/>
Lars Kaesberg <br/>
Niklas Bauer <br/>
Constantin Dalinghaus <br/>
Sebastian Kampen <br/>
</div>

## Introduction
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Work in Progress](https://img.shields.io/badge/Status-Work%20in%20Progress-red.svg)
![Black Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)

This repository is the official implementation of the Multitask BERT project for the Deep Learning for Natural Language
Processing course at the University of Göttingen.

A pretrained
BERT ([BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805))
model was used as the basis for our experiments. The model was fine-tuned on the three tasks using a multitask learning
approach. The model was trained on the three tasks simultaneously, with a single shared BERT encoder and three separate
task-specific classifiers.

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
python -u multitask_classifier.py --use_gpu
```

There are a lot of parameters that can be set. To see all of them, run `python multitask_classifier.py --help`. The most
important ones are:

| Parameter               | Description                                                                    |
|-------------------------|--------------------------------------------------------------------------------|
| `--additional_input`    | Activates the usage for POS and NER tags for the input of BERT                 |
| `--batch_size`          | Batch size.                                                                    |
| `--checkpoint`          | Path to a checkpoint to resume from.                                           |
| `--clip`                | Gradient clipping value.                                                       |
| `--epochs`              | Number of epochs.                                                              |
| `--hess_interval`       | Hessian update interval for SophiaH.                                           |
| `--hidden_dropout_prob` | Dropout probability for hidden layers.                                         |
| `--hpo_trials`          | Number of trials for hyperparameter optimization.                              |
| `--hpo`                 | Activate hyperparameter optimization.                                          |
| `--lr`                  | Learning rate.                                                                 |
| `--optimizer`           | Optimizer to use. Options are `AdamW` and `SophiaH`.            |
| `--option`              | Determines if BERT parameters are frozen (`pretrain`) or updated (`finetune`). |
| `--rho`                 | rho for SophiaH optimizer.                                                     |
| `--samples_per_epoch`   | Number of samples per epoch.                                                   |
| `--scheduler`           | Learning rate scheduler to use. Options are `plateau`, `cosine`, and `none`.   |
| `--unfreeze_interval`   | Number of epochs until the next BERT layer is unfrozen                         |
| `--use_gpu`             | Whether to use the GPU.                                                        |
| `--weight_decay`        | Weight decay for optimizer.                                                    |

> 📋 Describe how to train the models, with example commands on how to train the models in your paper, including the full
> training procedure and appropriate hyperparameters.

## Evaluation

The model is evaluated after each epoch on the validation set. The results are printed to the console and saved in
the `logdir` directory. The best model is saved in the `models` directory.

## Methodology

This section describes the methodology used in our experiments to extend the training of the multitask BERT model to the
three tasks of paraphrase identification, sentiment classification, and semantic textual similarity.

---

### POS and NER Tag Embeddings

Based on Bojanowski, et al. ([Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)), which showed that the
addition of subword information to word embeddings can improve performance on downstream tasks, we extended our approach
by incorporating Part-of-Speech (POS) and Named Entity Recognition (NER) tag embeddings into the input representation.
The primary goal was to investigate whether the inclusion of linguistic information could lead to improved performance
on the tasks.

#### Tagging

For the efficient and accurate tagging of POS and NER, we used the [spaCy](https://spacy.io/) library. The tagging
process occurs during data preprocessing, where each sentence is tokenized into individual words. The spaCy pipeline is
then invoked to annotate each word with its corresponding POS tag and NER label. The resulting tags and labels are
subsequently converted into embeddings.

To increase training efficiency, we implemented a caching mechanism where the computed tag embeddings were stored and
reused across multiple epochs.

#### Experimental Results

Contrary to our initial expectations, the inclusion of POS and NER tag embeddings did not yield the desired improvements
across the three tasks. Experimental results indicated that the performance either remained stagnant or even slightly
deteriorated compared to the baseline BERT model without tag embeddings.

#### Impact on Training Process

An additional observation was the notable increase in training time when incorporating POS and NER tag embeddings. This
extended training time was attributed to the additional computational overhead required for generating and embedding the
tags.

#### Conclusion

Although the integration of POS and NER tag embeddings initially seemed promising, our experiments showed that this
approach did not contribute to performance across tasks. The training process was noticeably slowed down by the
inclusion of tag embeddings.

As a result, we concluded that the benefits of incorporating POS and NER tags were not substantial enough to justify the
extended training time. Future research could explore alternative ways of effectively exploiting linguistic features
while minimising the associated computational overhead.

One possible explanation for the lack of performance improvements could be that the BERT model already encodes some
syntactic information in its word
embeddings. Hewitt and Manning ([A Structural Probe for Finding Syntax in Word Representations](https://aclanthology.org/N19-1419.pdf))
showed that some syntactic information is already encoded in the word embeddings of pretrained BERT models, which could
explain why the inclusion of POS and NER tags did not lead to performance improvements.

---

### Sophia

We implemented the Sophia (**S**econd-**o**rder Cli**p**ped Stoc**h**astic Opt**i**miz**a**tion) optimizer completly
from scratch, which is a second-order optimizer for language model pre-training. The paper promises convergence twice as
fast as AdamW and better generalisation performance. It uses a light weight estimate of the diagonal of the Hessian
matrix to approximate the curvature of the loss function. It also uses clipping to control the worst-case update size.
By only updating the Hessian estimate every few iterations, the overhead is negligible.

The optimizer was introduced recently in the
paper [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342).

#### Implementation

The paper describes the optimizer in detail, but does not provide any usable code. We implemented the optimizer from
scratch in PyTorch. The optimizer is implemented in the [`optimizer.py`](optimizer.py) file and can be used in the
multitask classifier by setting the `--optimizer` parameter.

There are two ways of estimating the Hessian. The first option is to use the Gauss-Newton-Bartlett approximation, which
is computed using an average over the minibatch gradients. However, this estimator requires the existence of a
multi-class classification problem from which to sample. This is not the case for some of our tasks, e.g. STS, which is
a regression task. The estimator is still implemented as `SophiaG`.

The second option is to use Hutchinson's unbiased estimator of the Hessian diagonal by sampling from a spherical
Gaussian distribution. This estimator is implemented as `SophiaH`. This estimator can be used for all tasks. It requires
a Hessian vector product, which is implemented in most modern deep learning frameworks, including PyTorch.

#### Convergence

While the implementation of this novel optimizer was a challenge, the results were not as promised. The optimizer did
not converge faster than AdamW, and the performance was comparable. This could be due to the fact that the optimizer was
designed for pre-training language models, which is a different task to ours.

A more recent paper studing different training algorithms for transformer-based language
models by Kaddour et al. ([No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models](https://arxiv.org/pdf/2307.06440.pdf))
comes to the conclusion that the training algorithm gains vanish with a fully decayed learning rate. They show
performance being about the same as the baseline (AdamW), which is what we observed.

---

### Data Augmentation

In light of emerging advancements in imitation learning—particularly the success of small language models in approximating the performance of larger, proprietary models (Taori et al., 2023)—this study investigates the efficacy of synthetic data for enhancing multitask classification algorithms. The focus of our proof-of-concept experiment is on sentiment classification, a task that presents significant challenges. We explore three methodologies for synthetic data generation: 1) constructing a language model from the ground up, 2) finetuning an existing language model, and 3) employing prompts for data generation.

#### De Novo Transformer-based Language Model

Constructing a transformer-based language model from scratch yielded data of suboptimal quality, characterized by out-of-distribution samples. The model manifested difficulties in assimilating the available training data, often resulting in incoherent outputs.

#### Finetuning GPT-2

We employed the GPT-2 medium model variant by OpenAI (Radford et al., 2018) and finetuned it using a constant learning rate on the sentiment classification training set. This finetuned model was then used to generate 100,000 training samples, an increase by an order of magnitude compared to the original dataset. While these samples were more contextually relevant compared to those from the first method, they still exhibited coherence issues to some extent.

#### Prompt-based Data Generation
For our third approach, we utilized a custom prompt to solicit synthetic data from ChatGPT-4:

```
For building a sentiment classifier, I need highly educational examples. I will provide you with some examples, please continue to give me examples on the same level of quality and average complexity. Please also respect the quirky formatting and return them as tab-seperated:

shuf -n 15 ids-sst-train.csv | awk -F'\t' '{print $3 "\t" $4}'
```
The data sampled by GPT4 is of the highest quality available. However, due to the closed-source nature of GPT4 and the limitations opposed on the chatGPT premium plan, only a limited quantity of this data was able to be acquired (~500 examples)

#### Results
None of the above methods improved the model performance beyond the capabilities of our best model. Notably, using 100.000 synthetic examples from GPT2, our model did not overfit to the train set, even after 30 epochs. It can be argued that the model may not be converged, however validation loss ceised to improve further so training was ended without full convergence achieved.

#### Disclaimer: Synthetic data
It is important to note that only the first method assures complete freedom from data contamination. GPT-2 and GPT-4 models were trained on undisclosed datasets, raising the potential issue of data overlap with our sentiment classification dataset. While it is improbable that these models would replicate specific test set examples, the caveat remains. Our exploration of these methods should be interpreted as an educational exercise, and we make no claims of superior model performance.

---

### More Details

#### Classifier

The design and selection of classifiers are crucial in multi-task learning, especially when the tasks are deeply
intertwined. The performance of one classifier can cascade its effects onto others, either enhancing the overall results
or, conversely, dragging them down. In our endeavor, we dedicated significant time to experimentation, aiming to ensure
not only the individual performance of each classifier but also their harmonious interaction within the multi-task
setup.

Some of the components of our multitask classifier are described in more detail below. Each classifier's architecture is
tailored to the unique characteristics of its task, enabling our multi-task learning framework to address multiple NLP
challenges simultaneously.

##### Attention Layer

The attention mechanism plays a major role in capturing and emphasizing salient information within the output embeddings
generated by the BERT model. We implemented
an `AttentionLayer` ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)) that accepts the last hidden state
of the BERT output and applies a weighted sum mechanism to enhance the importance of certain tokens while suppressing
others. This layer aids in creating a more focused representation of the input sentence, which is crucial for downstream
tasks.

##### Sentiment Analysis Classifier

This classifier architecture consists of several linear layers that refine the BERT embeddings into logits corresponding
to each sentiment class. These logits are then used to compute the predicted sentiment label. Achieving a balance here
was crucial, as any inefficiencies could potentially impact the overall performance of our multi-task framework.

##### Paraphrase Detection Classifier

The paraphrase detection classifier uses a two-step process. First, the BERT embeddings for each input sentence are
processed separately by a linear layer. We then compute the absolute difference and the absolute sum of these processed
embeddings. These two concatenated features are then fed through additional linear layers to generate logits for
paraphrase prediction. Iterative refinement was crucial here, ensuring that the classifier neither overshadowed nor was
overshadowed by the other tasks.

##### Semantic Textual Similarity Estimator

For the Semantic Textual Similarity task, our approach relies on cosine similarity. The BERT embeddings for the input
sentences are generated and then compared using cosine similarity. The resulting similarity score is scaled to range
between 0 and 5, providing an estimate of how semantically similar the two sentences are.

#### Layer Unfreeze

Layer unfreezing is a technique employed during fine-tuning large pre-trained models like BERT. The idea behind
this method is to gradually unfreeze layers of the model during the training process. Initially, the top layers are trained while the bottom layers are frozen. As training progresses, more layers are incrementally
unfrozen, allowing for deeper layers of the model to be adjusted.

One of the motivations to use layer unfreezing is to prevent *catastrophic forgetting*—a phenomenon where the model
rapidly forgets its previously learned representations when fine-tuned on a new
task ([Howard and Ruder](https://arxiv.org/abs/1801.06146)). By incrementally unfreezing the layers, the hope is to
preserve valuable pretrained representations in the earlier layers while allowing the model to adapt to the new task.

In our implementation, we saw a decrease in performance. One possible
reason for this could be the interaction between the layer thaw schedule and the learning rate scheduler (plateau). As the
learning rate scheduler reduced the learning rate, not all layers were yet unfrozen. This mismatch may have hindered
the model's ability to make effective adjustments to the newly unfrozen layers. As a result, the benefits expected from the
unfreezing layers may have been offset by this unintended interaction.

#### Mixture of Experts

#### Automatic Mixed Precision

The automatic mixed precision (AMP) feature of PyTorch was used to speed up training and reduce memory usage. This feature changes the precision of the model's weights and activations during training. The model was trained in `bfloat16` precision, which is a fast 16-bit floating point format. The AMP feature of PyTorch automatically casts the model parameters. This reduces the memory usage and speeds up training.

## Experiments

We used the default datasets provided for training and validation with no modifications.

The baseline for our comparisons includes most smaller improvements to the BERT model listed above. The baseline model is further described in the [Results](#results) section. The baseline model was trained for 10 epochs at 10.000 samples per epoch.

The models were trained and evaluated on the Grete cluster. The training was done on a single A100 GPU. The training time for the baseline model was approximately 2 hours.

We used [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to perform hyperparameter tuning. This allowed us to efficiently explore the hyperparameter space and find the best hyperparameters for our model. We used [Optuna](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html) to search the hyperparameter space and [AsyncHyperBandScheduler](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html) as the scheduler. The hyperparameters were searched for the whole model, not for each task individually. This was done to avoid overfitting to a single task. We searched for hyperparameters trying to minimize the overfitting of the model to the training data.

The trained models were evaluated on the validation set. The best model was selected based on the validation results ('dev'). The metrics used for the evaluation were accuracy only for paraphrase identification and sentiment classification, and Pearson correlation for semantic textual similarity.

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

## Comparisons

### [Paraphrase Identification](https://paperswithcode.com/paper/spanbert-improving-pre-training-by)

### [Sentiment Classification](https://paperswithcode.com/paper/fine-grained-sentiment-classification-using)

### [Semantic Textual Similarity](https://paperswithcode.com/paper/albert-a-lite-bert-for-self-supervised)

## PyTorch Profiler Results
<details>
  <summary>Click me</summary>
We utilized the `pytorch_profiler` integrated with TensorBoard to gain insights into the execution performance and resource utilization during our model's training on a GPU.

### Configuration Details:
- **Number of Workers:** 1
- **Device Type:** GPU

### GPU Summary:
- **Model:** NVIDIA A100-SXM4-80GB
- **Total Memory:** 79.19 GB
- **Compute Capability:** 8.0
- **GPU Utilization:** 64.35%
- **Estimated SM Efficiency:** 59.55%
- **Estimated Achieved Occupancy:** 47.89%

### Execution Breakdown:

| Category          | Time Duration (us) | Percentage (%) |
|-------------------|--------------------|----------------|
| Average Step Time | 2,199,623          | 100            |
| GPU Kernel        | 1,415,549          | 64.35          |
| Memcpy            | 3,064              | 0.14           |
| Memset            | 4,455              | 0.20           |
| CPU Execution     | 574,478            | 26.12          |
| Other             | 202,077            | 9.19           |
| DataLoader        | 0                  | 0              |
| Runtime           | 0                  | 0              |

### Insights:

The profiler results provide a detailed look into how the model's computations are distributed. A significant portion of the execution time (64.35%) is consumed by GPU kernel operations, indicating that the bulk of the computational heavy-lifting is done on the GPU. Meanwhile, CPU-related tasks took up about a quarter of the total execution time at 26.12%. Operations such as `Memcpy` and `Memset` have minimal impact on the overall performance.

Given the GPU utilization rate of 64.35% and the estimated SM efficiency, there may be room for optimization in the future. Enhancing kernel functions or restructuring model operations might lead to improvements in SM efficiency and occupancy, potentially boosting overall performance.

</details>

## Contributors

| Lars Kaesberg    | Niklas Bauer | Constantin Dalinghaus |
|------------------|--------------|-----------------------|
| Tagging          | Sophia Optimizer       | Synthetic Data        |
| Layer Unfreeze   | Hyperparameter Tuning          |     |
| Classifier Model | Repository   |                       |

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
