from pprint import pformat
import time, random, numpy as np, argparse, sys, re, os
from datetime import datetime
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bert import BertModel
from AttentionLayer import AttentionLayer
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask

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
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        # Common layers
        self.activation = nn.ReLU()
        self.cosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.attention_layer = AttentionLayer(config.hidden_size)

        # Similarity task
        sts_hidden_size = 256
        self.similarity_representation1 = nn.Linear(config.hidden_size, config.hidden_size)

        # Sentiment task
        self.linear_layer = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)

        # Paraphrase task
        para_hidden_size = 256
        self.para_representation1 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        result = self.bert(input_ids, attention_mask)
        attention_result = self.attention_layer(result["last_hidden_state"])
        return attention_result

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        return self.linear_layer(self.forward(input_ids, attention_mask))

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        """

        bert_embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        bert_embeddings_2 = self.forward(input_ids_2, attention_mask_2)

        embeddings_1_representation = self.activation(
            self.para_representation1(bert_embeddings_1)) + bert_embeddings_1
        embeddings_2_representation = self.activation(
            self.para_representation1(bert_embeddings_2)) + bert_embeddings_2

        output = self.cosineSimilarity(embeddings_1_representation, embeddings_2_representation)

        return output

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        """

        bert_embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        bert_embeddings_2 = self.forward(input_ids_2, attention_mask_2)

        embeddings_1_representation = self.activation(
            self.similarity_representation1(bert_embeddings_1)) + bert_embeddings_1
        embeddings_2_representation = self.activation(
            self.similarity_representation1(bert_embeddings_2)) + bert_embeddings_2

        output = self.cosineSimilarity(embeddings_1_representation, embeddings_2_representation)
        return output


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
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
    optimizer.load_state_dict(save_info['optim'])

    # Retrieve other saved information
    # device is alredy set at this point
    args = save_info['args']
    args.use_gpu = use_gpu
    config = save_info['model_config']
    random.setstate(save_info['system_rng'])
    np.random.set_state(save_info['numpy_rng'])
    torch.random.set_rng_state(save_info['torch_rng'])

    print(f"Loaded the model from {filepath}")

    return model, optimizer, args, config


## Currently only trains on sst dataset
def train_multitask(args):
    loss_sst_idx_value = 0
    loss_sts_idx_value = 0
    loss_para_idx_value = 0

    train_all_datasets = True
    n_datasets = args.sst + args.sts + args.para
    if args.sst or args.sts or args.para:
        train_all_datasets = False
    if n_datasets == 0:
        n_datasets = 3

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train,
                                                                                      args.para_train,
                                                                                      args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev,
                                                                                args.sts_dev, split='train')

    sst_train_dataloader = None
    sst_dev_dataloader = None
    para_train_dataloader = None
    para_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    total_num_batches = 0
    if train_all_datasets or args.sst:
        sst_train_data = SentenceClassificationDataset(sst_train_data, args, override_length=args.samples_per_epoch)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=sst_train_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)
        total_num_batches += len(sst_train_dataloader)

    if train_all_datasets or args.para:
        para_train_data = SentencePairDataset(para_train_data, args, override_length=args.samples_per_epoch)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                           collate_fn=para_train_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)
        total_num_batches += len(para_train_dataloader)

    if train_all_datasets or args.sts:
        sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True,
                                             override_length=args.samples_per_epoch)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=sts_train_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)
    total_num_batches += len(sts_train_dataloader)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'local_files_only': args.local_files_only}

    config = SimpleNamespace(**config)

    print("Multitask BERT model:", file=sys.stderr)
    print(pformat(vars(args)), file=sys.stderr)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5, verbose=True)
    elif args.scheduler == 'cosine':
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
    writer = SummaryWriter(log_dir=args.logdir + "/multitask_classifier/" + name)

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for sts, para, sst in tqdm(zip(sts_train_dataloader, para_train_dataloader, sst_train_dataloader),
                                   total=len(sts_train_dataloader), desc=f"train-{epoch}", disable=TQDM_DISABLE):
            optimizer.zero_grad()

            # Train on STS dataset
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                sts['token_ids_1'], sts['attention_mask_1'], sts['token_ids_2'], sts['attention_mask_2'],
                sts['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)

            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)

            b_labels = b_labels.to(device)

            logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            b_labels = b_labels.to(torch.float32)
            sts_loss = F.mse_loss(logits, b_labels.view(-1))

            # Train on PARAPHRASE dataset
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                para['token_ids_1'], para['attention_mask_1'], para['token_ids_2'], para['attention_mask_2'],
                para['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)

            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)

            b_labels = b_labels.to(device)

            logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            b_labels = b_labels.to(torch.float32)
            para_loss = F.mse_loss(logits, b_labels.view(-1))

            # Train on SST dataset
            b_ids, b_mask, b_labels = (sst['token_ids'],
                                       sst['attention_mask'], sst['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            sst_loss = F.cross_entropy(logits, b_labels.view(-1))

            full_loss = sts_loss + para_loss + sst_loss
            full_loss.backward()

            train_loss += full_loss.item()
            num_batches += 1

            optimizer.step()

            # print(full_loss.item())
            # print("DONE")

        train_loss = train_loss / num_batches
        writer.add_scalar("Loss/Epochs", train_loss, epoch)

        para_train_acc, _, _, sst_train_acc, _, _, sts_train_acc, _, _ = model_eval_multitask(sst_train_dataloader,
                                                                                              para_train_dataloader,
                                                                                              sts_train_dataloader,
                                                                                              model, device)

        para_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_acc, _, _ = model_eval_multitask(sst_dev_dataloader,
                                                                                        para_dev_dataloader,
                                                                                        sts_dev_dataloader, model,
                                                                                        device)
        if args.para:
            writer.add_scalar("para_acc/train/Epochs", para_train_acc, epoch)
            writer.add_scalar("para_acc/dev/Epochs", para_dev_acc, epoch)
        if args.sst:
            writer.add_scalar("sst_acc/train/Epochs", sst_train_acc, epoch)
            writer.add_scalar("sst_acc/dev/Epochs", sst_dev_acc, epoch)
        if args.sts:
            writer.add_scalar("sts_acc/train/Epochs", sts_train_acc, epoch)
            writer.add_scalar("sts_acc/dev/Epochs", sts_dev_acc, epoch)

        if para_dev_acc > best_dev_acc_para and sst_dev_acc > best_dev_acc_sst and sts_dev_acc > best_dev_acc_sts:
            best_dev_acc_para = para_dev_acc
            best_dev_acc_sst = sst_dev_acc
            best_dev_acc_sts = sts_dev_acc
            save_model(model, optimizer, args, config, args.filepath)
        train_acc = sst_train_acc + para_train_acc + sts_train_acc
        dev_acc = sst_dev_acc + para_dev_acc + sts_dev_acc

        if args.scheduler == 'plateau':
            scheduler.step(dev_acc)

        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("acc/train/Epochs", train_acc, epoch)
        writer.add_scalar("acc/dev/Epochs", dev_acc, epoch)
        print(
            f"Epoch {epoch}: train loss :: {train_loss :.3f}, combined train acc :: {train_acc :.3f}, combined dev acc :: {dev_acc :.3f}")


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
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

    parser.add_argument("--samples_per_epoch", type=int, default=None)

    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sts", action='store_true')
    parser.add_argument("--sst", action='store_true')
    parser.add_argument("--para", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--logdir", type=str, default="logdir")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--local_files_only", action='store_true')
    parser.add_argument("--scheduler", type=str, default="plateau", choices=('plateau', 'cosine', 'none'))

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # TODO Add optimizer after Sophia merge
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-{args.scheduler}-multitask.pt'  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
