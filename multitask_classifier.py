import argparse
import math
import os
import random
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime
from pprint import pformat
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_optimizer import SophiaH
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from AttentionLayer import AttentionLayer
from bert import BertModel
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW

TQDM_DISABLE = False


# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    """
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        if config.unfreeze_interval:
            if config.option == "pretrain":
                for name, param in self.bert.named_parameters():
                    if not name.startswith("bert_layers"):
                        continue
                    param.requires_grad = False
            else:
                print("Unfreeze used in finetune mode, ignoring")

        self.use_additional_input = config.additional_input

        self.attention_layer = AttentionLayer(config.hidden_size)

        # SENTIMENT
        self.sentiment_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.sentiment_linear1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.sentiment_linear2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.sentiment_linear_out = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)

        # PARAPHRASE
        self.paraphrase_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.paraphrase_linear1 = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.paraphrase_linear2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.paraphrase_out = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        "Takes a batch of sentences and produces embeddings for them."
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        result = self.bert(input_ids, attention_mask, self.use_additional_input)
        attention_result = self.attention_layer(result["last_hidden_state"])
        return attention_result

    def predict_sentiment(self, input_ids, attention_mask):
        """Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        """
        bert_embedding = self.forward(input_ids, attention_mask)
        logits = F.relu(self.sentiment_linear(bert_embedding))
        logits = F.relu(self.sentiment_linear1(logits))
        logits = F.relu(self.sentiment_linear2(logits))
        logits = self.sentiment_linear_out(logits)
        return logits

    def predict_paraphrase_train(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Given a batch of pairs of sentences, outputs logits for predicting whether they are paraphrases.
        """
        bert_embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        bert_embeddings_2 = self.forward(input_ids_2, attention_mask_2)

        combined_bert_embeddings_1 = self.paraphrase_linear(bert_embeddings_1)
        combined_bert_embeddings_2 = self.paraphrase_linear(bert_embeddings_2)

        # Calculate absolute difference and sum of combined embeddings
        abs_diff = torch.abs(combined_bert_embeddings_1 - combined_bert_embeddings_2)
        abs_sum = torch.abs(combined_bert_embeddings_1 + combined_bert_embeddings_2)

        # Concatenate the absolute difference and sum
        concatenated_features = torch.cat((abs_diff, abs_sum), dim=1)

        # Apply linear layers to obtain logits for both "yes" and "no" predictions
        logits = F.relu(self.paraphrase_linear1(concatenated_features))
        logits = F.relu(self.paraphrase_linear2(logits))
        logits = self.paraphrase_out(logits)

        return logits

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        logits = self.predict_paraphrase_train(
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
        )
        return logits.argmax(dim=-1)

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        """

        bert_embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        bert_embeddings_2 = self.forward(input_ids_2, attention_mask_2)

        diff = torch.cosine_similarity(bert_embeddings_1, bert_embeddings_2)
        return diff * 5


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def load_model(filepath, model, optimizer, use_gpu):
    # Check if the file exists
    if not os.path.isfile(filepath):
        raise ValueError(f"File {filepath} does not exist")

    save_info = torch.load(
        filepath,
        map_location=torch.device("cuda") if use_gpu else torch.device("cpu"),
    )

    # Load model state
    model.load_state_dict(save_info["model"])

    # Load optimizer state
    optimizer.load_state_dict(save_info["optim"])

    # Retrieve other saved information
    # device is alredy set at this point
    args = save_info["args"]
    args.use_gpu = use_gpu
    config = save_info["model_config"]
    random.setstate(save_info["system_rng"])
    np.random.set_state(save_info["numpy_rng"])
    torch.random.set_rng_state(save_info["torch_rng"])

    print(f"Loaded the model from {filepath}")

    return model, optimizer, args, config


## Currently only trains on sst dataset
def train_multitask(args):
    # Ray: Need to convert args from a dict to a Namespace
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    train_all_datasets = True
    n_datasets = args.sst + args.sts + args.para
    if args.sst or args.sts or args.para:
        train_all_datasets = False
    if n_datasets == 0:
        n_datasets = 3

    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split="train"
    )
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split="train"
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    para_train_dataloader = None
    para_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    total_num_batches = 0
    # if train_all_datasets or args.sst:
    sst_train_data = SentenceClassificationDataset(
        sst_train_data, args, override_length=args.samples_per_epoch
    )
    sst_dev_data = SentenceClassificationDataset(
        sst_dev_data, args, override_length=10 if args.smoketest else None
    )

    sst_train_dataloader = DataLoader(
        sst_train_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sst_train_data.collate_fn,
        num_workers=2,
    )
    sst_dev_dataloader = DataLoader(
        sst_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sst_dev_data.collate_fn,
        num_workers=2,
    )

    # if train_all_datasets or args.para:
    para_train_data = SentencePairDataset(
        para_train_data, args, override_length=args.samples_per_epoch
    )
    para_dev_data = SentencePairDataset(
        para_dev_data, args, override_length=10 if args.smoketest else None
    )

    para_train_dataloader = DataLoader(
        para_train_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn,
        num_workers=2,
    )
    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn,
        num_workers=2,
    )

    # if train_all_datasets or args.sts:
    sts_train_data = SentencePairDataset(
        sts_train_data, args, isRegression=True, override_length=args.samples_per_epoch
    )
    sts_dev_data = SentencePairDataset(
        sts_dev_data, args, isRegression=True, override_length=10 if args.smoketest else None
    )

    sts_train_dataloader = DataLoader(
        sts_train_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sts_train_data.collate_fn,
        num_workers=2,
    )
    sts_dev_dataloader = DataLoader(
        sts_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sts_dev_data.collate_fn,
        num_workers=2,
    )

    total_num_batches += math.ceil(args.samples_per_epoch / args.batch_size)

    # Init model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "num_labels": num_labels,
        "additional_input": args.additional_input,
        "hidden_size": 768,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
        "unfreeze_interval": args.unfreeze_interval,
    }

    config = SimpleNamespace(**config)

    # Print model configuration
    separator = "=" * 60
    print(separator)
    print("    Multitask BERT Model Configuration")
    print(separator)
    filtered_vars = {
        k: v for k, v in vars(args).items() if "csv" not in str(v)
    }  # Filter out csv files
    print(pformat(filtered_vars))
    print("-" * 60)

    # Print Git info
    branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()[:8]
    is_modified = (
        "(!)"
        if subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
        else ""
    )

    # Print Git info
    print(f"Git Branch: {branch}")
    print(f"Git Hash: {commit} {is_modified}")
    print("-" * 60)  # Adjust as needed
    print(f"Command: {' '.join(sys.argv)}")
    print(separator)

    model = MultitaskBERT(config)

    device = torch.device("cpu")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs", file=sys.stderr)
            model = nn.DataParallel(model)

    model = model.to(device)

    lr = args.lr
    hess_interval = args.hess_interval
    ctx = (
        nullcontext()
        if not args.use_gpu
        else torch.amp.autocast(device_type="cuda", dtype=torch.float32)
    )

    if args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sophiah":
        # TODO: Tune this further, https://github.com/Liuhong99/Sophia#hyper-parameter-tuning
        optimizer = SophiaH(
            model.parameters(),
            lr=lr,
            betas=(0.985, 0.99),
            weight_decay=args.weight_decay,
            eps=1e-12,
            p=args.rho,
            update_period=hess_interval,
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, 1)
    else:
        scheduler = None

    best_dev_acc_para = 0
    best_dev_acc_sst = 0
    best_dev_acc_sts = 0

    if args.checkpoint:
        # New args is not used!
        model, optimizer, _, config = load_model(args.checkpoint, model, optimizer, args.use_gpu)

    name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.epochs}-{type(optimizer).__name__}-{lr}-{args.scheduler}"
    path = (
        args.logdir
        + "/multitask_classifier/"
        + (f"{args.tensorboard_subfolder}/" if args.tensorboard_subfolder else "")
        + name
    )
    writer = SummaryWriter(log_dir=path)

    writer.add_hparams(vars(args), {}, run_name="hparams")

    if args.profiler:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(path + "_profiler"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        num_layers = model.bert.config.num_hidden_layers

        # Unfreeze the layers
        unfreezed = set()
        if args.unfreeze_interval and args.option == "pretrain":
            for name, param in model.bert.named_parameters():
                if not name.startswith("bert_layers"):
                    continue
                layer_num = int(name.split(".")[1])
                unfreeze_up_to = num_layers - epoch // args.unfreeze_interval
                if layer_num >= unfreeze_up_to:
                    unfreezed.add(layer_num)
                    param.requires_grad = True  # Unfreeze the layer

        if len(unfreezed) > 0:
            print(f"Unfreezed BERT layers: {unfreezed}", file=sys.stderr)

        for sts, para, sst in tqdm(
            zip(sts_train_dataloader, para_train_dataloader, sst_train_dataloader),
            total=total_num_batches,
            desc=f"train-{epoch}",
            disable=TQDM_DISABLE,
        ):
            optimizer.zero_grad(set_to_none=True)
            sts_loss, para_loss, sst_loss = 0, 0, 0

            # Train on STS dataset
            if train_all_datasets or args.sts:
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    sts["token_ids_1"],
                    sts["attention_mask_1"],
                    sts["token_ids_2"],
                    sts["attention_mask_2"],
                    sts["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)

                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)

                b_labels = b_labels.to(device)

                with ctx:
                    logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                    b_labels = b_labels.to(torch.float32)
                    sts_loss = F.mse_loss(logits, b_labels.view(-1))

            # Train on PARAPHRASE dataset
            if train_all_datasets or args.para:
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    para["token_ids_1"],
                    para["attention_mask_1"],
                    para["token_ids_2"],
                    para["attention_mask_2"],
                    para["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)

                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)

                b_labels = b_labels.to(device)

                with ctx:
                    logits = model.predict_paraphrase_train(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                    para_loss = F.cross_entropy(logits, b_labels.view(-1))

            # Train on SST dataset
            if train_all_datasets or args.sst:
                b_ids, b_mask, b_labels = (sst["token_ids"], sst["attention_mask"], sst["labels"])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                with ctx:
                    logits = model.predict_sentiment(b_ids, b_mask)
                    sst_loss = F.cross_entropy(logits, b_labels.view(-1))

            # Combined Loss
            # Can also weight the losses
            full_loss = sts_loss + para_loss + sst_loss
            full_loss.backward(create_graph=True if args.optimizer == "sophiah" else False)

            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # Update the parameters
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss += full_loss.item()
            num_batches += 1

            if args.scheduler == "cosine":
                # Potentially update the scheduler once per epoch instead
                scheduler.step(epoch + num_batches / total_num_batches)

            writer.add_scalar(
                "Loss/Minibatches", full_loss.item(), num_batches + epoch * total_num_batches
            )

            if args.profiler:
                prof.step()

        train_loss = train_loss / num_batches
        writer.add_scalar("Loss/Epochs", train_loss, epoch)

        para_train_acc, _, _, sst_train_acc, _, _, sts_train_acc, _, _ = model_eval_multitask(
            sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device
        )

        para_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_acc, _, _ = model_eval_multitask(
            sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device
        )

        writer.add_scalar("para_acc/train/Epochs", para_train_acc, epoch)
        writer.add_scalar("para_acc/dev/Epochs", para_dev_acc, epoch)
        writer.add_scalar("sst_acc/train/Epochs", sst_train_acc, epoch)
        writer.add_scalar("sst_acc/dev/Epochs", sst_dev_acc, epoch)
        writer.add_scalar("sts_acc/train/Epochs", sts_train_acc, epoch)
        writer.add_scalar("sts_acc/dev/Epochs", sts_dev_acc, epoch)

        if (
            para_dev_acc > best_dev_acc_para
            and sst_dev_acc > best_dev_acc_sst
            and sts_dev_acc > best_dev_acc_sts
        ):
            best_dev_acc_para = para_dev_acc
            best_dev_acc_sst = sst_dev_acc
            best_dev_acc_sts = sts_dev_acc
            save_model(model, optimizer, args, config, args.filepath)
        train_acc = sst_train_acc + para_train_acc + sts_train_acc
        dev_acc = sst_dev_acc + para_dev_acc + sts_dev_acc

        # Report to Ray Tune
        if args.hpo:
            session.report({"mean_accuracy": dev_acc / 3})

        if args.scheduler == "plateau":
            scheduler.step(dev_acc)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("acc/train/Epochs", train_acc, epoch)
        writer.add_scalar("acc/dev/Epochs", dev_acc, epoch)
        print(
            f"Epoch {epoch}: train loss :: {train_loss :.3f}, combined train acc :: {train_acc :.3f}, combined dev acc :: {dev_acc :.3f}"
        )


def test_model(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )

    parser.add_argument("--samples_per_epoch", type=int, default=30000)
    parser.add_argument("--unfreeze_interval", type=int, default=None)
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument("--additional_input", action="store_true")

    parser.add_argument("--profiler", action="store_true")

    parser.add_argument("--sts", action="store_true")
    parser.add_argument("--sst", action="store_true")
    parser.add_argument("--para", action="store_true")

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--logdir", type=str, default="logdir")

    # hyper parameters
    parser.add_argument("--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--clip", type=float, default=1.0, help="value used gradient clipping")

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=("adamw", "sophiah"),
        default="adamw",
    )
    parser.add_argument("--rho", type=float, default=0.05, help="rho for SophiaH optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--hess_interval", type=int, default=10, help="Hessian update interval for SophiaH"
    )

    args, _ = parser.parse_known_args()

    # TODO: Possibly change defaults based on optimizer
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-5 * (1 / args.rho if args.optimizer == "sophiah" else 1)
        if args.option == "finetune"
        else 1e-3 * (1 / args.rho if args.optimizer == "sophiah" else 1),
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tensorboard_subfolder", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--scheduler", type=str, default="plateau", choices=("plateau", "cosine", "none")
    )

    parser.add_argument("--hpo", action="store_true", help="Activate hyperparameter optimization")
    parser.add_argument("--smoketest", action="store_true", help="Run a smoke test")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"{args.option}-{args.epochs}-{args.lr}-{args.optimizer}-{args.scheduler}-multitask.pt"  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility

    # Ray Tune
    if args.hpo:
        import ray
        from ray import air, tune
        from ray.air import session
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.optuna import OptunaSearch

        config = vars(args)
        tune_config = {
            "lr": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([1, 2, 4]),
        }
        config.update(tune_config)

        # Scheduler: Async Hyperband
        scheduler = ASHAScheduler(
            max_t=args.epochs,
            grace_period=1,
            reduction_factor=2,
        )

        # Search Algorithm: Optuna
        algo = OptunaSearch(metric="mean_accuracy", mode="max")

        ray.init(log_to_driver=False)  # Don't print logs to console

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train_multitask),
                resources={"cpu": 4, "gpu": torch.cuda.device_count()},
            ),
            tune_config=tune.TuneConfig(
                metric="mean_accuracy",
                mode="max",
                num_samples=10,  # Number of trials
                scheduler=scheduler,
                search_alg=algo,
                chdir_to_trial_dir=False,  # Still access local files
                max_concurrent_trials=4,  # Number of trials to run concurrently
            ),
            run_config=air.RunConfig(log_to_file="std.log", verbose=2),  # Don't spam CLI
            param_space=config,
        )

        results = tuner.fit()
        best_result = results.get_best_result("mean_accuracy", "max")

        separator = "=" * 60
        print(separator)
        print("    Best Multitask BERT Model Configuration")
        print(separator)
        filtered_vars = {
            k: v for k, v in best_result.config.items() if "csv" not in str(v)
        }  # Filter out csv files
        print(pformat(filtered_vars))
        print("-" * 60)
        print("Best mean_accuracy: ", best_result.metrics["mean_accuracy"])
    else:
        try:
            train_multitask(args)
            if not args.smoketest:
                test_model(args)
        except KeyboardInterrupt:
            print("Keyboard interrupt.")
