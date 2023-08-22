import argparse
import csv
import random
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from AttentionLayer import AttentionLayer
from bert import BertModel
from optimizer import AdamW
from optimizer import SophiaG
from tokenizer import BertTokenizer

# change it with respect to the original model

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


class BertSentimentClassifier(torch.nn.Module):
    """
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    """

    def __init__(self, config):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

        # Pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # linear layer to get logits
        self.attention_layer = AttentionLayer(config.hidden_size)
        self.linear_layer = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and returns logits for sentiment classes"""
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: you should consider what is the appropriate output to return given that
        # the training loop currently uses F.cross_entropy as the loss function.
        # Cross entropy already has a softmax therefore this should be okay

        # No Dropout because it is the last layer before softmax, else worse performance
        result = self.bert(input_ids, attention_mask)
        attention_result = self.attention_layer(result["last_hidden_state"])
        return self.linear_layer(attention_result)


class SentimentDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sents": sents,
            "sent_ids": sent_ids,
        }

        return batched_data


class SentimentTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "sents": sents,
            "sent_ids": sent_ids,
        }

        return batched_data


# Load the data: a list of (sentence, label)
def load_data(filename, flag="train"):
    num_labels = {}
    data = []
    if flag == "test":
        with open(filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent = record["sentence"].lower().strip()
                sent_id = record["id"].lower().strip()
                data.append((sent, sent_id))
    else:
        with open(filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent = record["sentence"].lower().strip()
                sent_id = record["id"].lower().strip()
                label = int(record["sentiment"].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label, sent_id))
        print(f"load {len(data)} data from {filename}")

    if flag == "train":
        return data, len(num_labels)
    else:
        return data


# Evaluate the model for accuracy.
def model_eval(dataloader, model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f"eval", disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = (
            batch["token_ids"],
            batch["attention_mask"],
            batch["labels"],
            batch["sents"],
            batch["sent_ids"],
        )

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids


def model_test_eval(dataloader, model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f"eval", disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sents, b_sent_ids = (
            batch["token_ids"],
            batch["attention_mask"],
            batch["sents"],
            batch["sent_ids"],
        )

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    return y_pred, sents, sent_ids


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


def train(args):
    loss_idx_value = 0

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load data
    # Create the data and its corresponding datasets and dataloader
    train_data, num_labels = load_data(args.train, "train")
    dev_data = load_data(args.dev, "valid")

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn
    )

    # Init model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "num_labels": num_labels,
        "hidden_size": 768,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
    }

    config = SimpleNamespace(**config)
    ctx = (
        nullcontext()
        if not args.use_gpu
        else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    )

    model = BertSentimentClassifier(config)
    model = model.to(device)

    lr = args.lr
    if args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif args.optimizer == "sophiag":
        optimizer = SophiaG(
            model.parameters(), lr=lr, eps=1e-12, rho=0.03, betas=(0.985, 0.99), weight_decay=2e-1
        )

    hess_interval = 10
    iter_num = 0

    best_dev_acc = 0

    # Initialize the tensorboard writer
    name = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-lr={lr}-optimizer={type(optimizer).__name__}"
    )
    writer = SummaryWriter(log_dir=args.logdir + "/classifier/" + name)

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(train_dataloader, desc=f"train-{epoch}", disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch["token_ids"], batch["attention_mask"], batch["labels"])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            with ctx:
                logits = model(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1))
            loss.backward()

            # Potentially: Clip gradients using
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # Check if we use the Sophia Optimizer
            if (
                hasattr(optimizer, "update_hessian")
                and iter_num % hess_interval == hess_interval - 1
            ):
                # Update the Hessian EMA
                with ctx:
                    logits = model(b_ids, b_mask)
                    samp_dist = torch.distributions.Categorical(logits=logits)
                    y_sample = samp_dist.sample()
                    loss_sampled = F.cross_entropy(logits, y_sample.view(-1))
                loss_sampled.backward()

                # Potentially: Clip gradients using
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.update_hessian(bs=args.batch_size)

                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()

            writer.add_scalar("Loss/Minibatches", loss.item(), loss_idx_value)
            loss_idx_value += 1
            num_batches += 1
            iter_num += 1

        train_loss = train_loss / (num_batches)
        writer.add_scalar("Loss/Epochs", train_loss, epoch)

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        writer.add_scalar("Accuracy/train/Epochs", train_acc, epoch)
        writer.add_scalar("F1_score/train/Epochs", train_f1, epoch)

        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)
        writer.add_scalar("Accuracy/dev/Epochs", dev_acc, epoch)
        writer.add_scalar("F1_score/dev/Epochs", dev_f1, epoch)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(
            f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}"
        )
    writer.close()


def test(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]
        model = BertSentimentClassifier(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"load model from {args.filepath}")

        dev_data = load_data(args.dev, "valid")
        dev_dataset = SentimentDataset(dev_data, args)
        dev_dataloader = DataLoader(
            dev_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=dev_dataset.collate_fn,
        )

        test_data = load_data(args.test, "test")
        test_dataset = SentimentTestDataset(test_data, args)
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=test_dataset.collate_fn,
        )

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(
            dev_dataloader, model, device
        )
        print("DONE DEV")
        test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
        print("DONE Test")
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sent_ids, dev_pred):
                f.write(f"{p} , {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sent_ids, test_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument("--logdir", type=str, default="logdir")
    parser.add_argument("--dev_out", type=str, default="sst-dev-out.csv")
    parser.add_argument("--test_out", type=str, default="sst-test-out.csv")

    parser.add_argument("--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--local_files_only", action="store_true")

    args, _ = parser.parse_known_args()

    # TODO: Possibly change defaults based on optimizer
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-5 if args.option == "finetune" else 1e-3,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    # args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt'

    print("Training Sentiment Classifier on SST...")
    config = SimpleNamespace(
        filepath="sst-classifier.pt",
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train="data/ids-sst-train.csv",
        dev="data/ids-sst-dev.csv",
        test="data/ids-sst-test-student.csv",
        option=args.option,
        dev_out="predictions/" + args.option + "-sst-dev-out.csv",
        test_out="predictions/" + args.option + "-sst-test-out.csv",
        logdir=args.logdir,
        optimizer=args.optimizer,
        local_files_only=args.local_files_only,
    )

    train(config)

    print("Evaluating on SST...")
    test(config)
