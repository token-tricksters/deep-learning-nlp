#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions to load the 3 datasets we're using.

You should only need to call load_multitask_data to get the training and dev examples
to train your model.
'''

import csv

import spacy
import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer
from random import randrange
import random


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


def get_pos_ner_tags(additional_input, token_ids, nlp, pos_tag_vocab, ner_tag_vocab):
    if additional_input:
        all_pos_tags = []
        all_ner_tags = []
        for sequence_id in token_ids:
            # Convert input_ids to tokens using the BERT tokenizer
            tokens = self.tokenizer.convert_ids_to_tokens(sequence_id.tolist())

            # Convert tokens to strings
            token_strings = [token if token != '[PAD]' else ' ' for token in tokens]

            # Create a Doc object from the list of tokens
            doc = spacy.tokens.Doc(nlp.vocab, words=token_strings)

            nlp.get_pipe("tok2vec")(doc)
            nlp.get_pipe("tagger")(doc)
            nlp.get_pipe("parser")(doc)
            nlp.get_pipe("ner")(doc)
            sequence_pos_indices = [pos_tag_vocab.get(tag.tag_, 0) for tag in doc]
            sequence_ner_indices = [ner_tag_vocab.get(tag.ent_type_, 0) for tag in doc]

            all_pos_tags.append(sequence_pos_indices)
            all_ner_tags.append(sequence_ner_indices)

        pos_tags_ids = torch.tensor(all_pos_tags, dtype=torch.long, device=token_ids.device)

        ner_tags_ids = torch.tensor(all_ner_tags, dtype=torch.long, device=token_ids.device)
    else:
        pos_tags_ids = torch.zeros(token_ids.shape, dtype=torch.long, device=token_ids.device)
        ner_tags_ids = torch.zeros(token_ids.shape, dtype=torch.long, device=token_ids.device)
    return pos_tags_ids, ner_tags_ids


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args, override_length=None, additional_input=False):
        self.override_length = override_length
        self.dataset = dataset
        self.p = args
        self.additional_input = additional_input

        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")

        pos_tags_spacy = self.nlp.get_pipe("tagger").labels
        ner_tags_spacy = self.nlp.get_pipe("ner").labels

        # Create a vocabulary dictionary for tags
        self.pos_tag_vocab = {tag: index + 1 for index, tag in enumerate(pos_tags_spacy)}
        self.ner_tag_vocab = {tag: index + 1 for index, tag in enumerate(ner_tags_spacy)}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)

    def real_len(self):
        return len(self.dataset)

    def __len__(self):
        if self.override_length is None:
            return self.real_len()
        return self.override_length

    def __getitem__(self, idx):
        if self.override_length is not None:
            return random.choice(self.dataset)

        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)
        pos_tags_ids, ner_tags_ids = get_pos_ner_tags(self.additional_input, token_ids, self.nlp, self.pos_tag_vocab,
                                                      self.ner_tag_vocab)

        return token_ids, attention_mask, labels, sents, sent_ids, pos_tags_ids, ner_tags_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids, pos_tags_ids, ner_tags_ids = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sents': sents,
            'sent_ids': sent_ids,
            'pos_tag_ids': pos_tags_ids,
            'ner_tag_ids': ner_tags_ids
        }

        return batched_data


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args, override_length=None, additional_input=False):
        self.override_length = override_length
        self.dataset = dataset
        self.p = args

        self.additional_input = additional_input

        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")

        pos_tags_spacy = self.nlp.get_pipe("tagger").labels
        ner_tags_spacy = self.nlp.get_pipe("ner").labels

        # Create a vocabulary dictionary for tags
        self.pos_tag_vocab = {tag: index + 1 for index, tag in enumerate(pos_tags_spacy)}
        self.ner_tag_vocab = {tag: index + 1 for index, tag in enumerate(ner_tags_spacy)}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)

    def real_len(self):
        return len(self.dataset)

    def __len__(self):
        if self.override_length is None:
            return self.real_len()
        return self.override_length

    def __getitem__(self, idx):
        if self.override_length is not None:
            return random.choice(self.dataset)

        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        pos_tags_ids, ner_tags_ids = get_pos_ner_tags(self.additional_input, token_ids, self.nlp, self.pos_tag_vocab,
                                                      self.ner_tag_vocab)

        return token_ids, attention_mask, sents, sent_ids, pos_tags_ids, ner_tags_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids, pos_tags_ids, ner_tags_ids = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sents': sents,
            'sent_ids': sent_ids,
            'pos_tag_ids': pos_tags_ids,
            'ner_tag_ids': ner_tags_ids
        }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False, override_length=None, additional_input=False):
        self.override_length = override_length
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression

        self.additional_input = additional_input

        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")

        pos_tags_spacy = self.nlp.get_pipe("tagger").labels
        ner_tags_spacy = self.nlp.get_pipe("ner").labels

        # Create a vocabulary dictionary for tags
        self.pos_tag_vocab = {tag: index + 1 for index, tag in enumerate(pos_tags_spacy)}
        self.ner_tag_vocab = {tag: index + 1 for index, tag in enumerate(ner_tags_spacy)}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)

    def real_len(self):
        return len(self.dataset)

    def __len__(self):
        if self.override_length is None:
            return self.real_len()
        return self.override_length

    def __getitem__(self, idx):
        if self.override_length is not None:
            return random.choice(self.dataset)

        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        pos_tags_ids, ner_tags_ids = get_pos_ner_tags(self.additional_input, token_ids, self.nlp, self.pos_tag_vocab,
                                                      self.ner_tag_vocab)

        pos_tags_ids2, ner_tags_ids2 = get_pos_ner_tags(self.additional_input, token_ids2, self.nlp, self.pos_tag_vocab,
                                                        self.ner_tag_vocab)

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                labels, sent_ids, pos_tags_ids, ner_tags_ids, pos_tags_ids2, ner_tags_ids2)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         labels, sent_ids, pos_tags_ids, ner_tags_ids, pos_tags_ids2, ner_tags_ids2) = self.pad_data(all_data)

        batched_data = {
            'token_ids_1': token_ids,
            'token_type_ids_1': token_type_ids,
            'attention_mask_1': attention_mask,
            'token_ids_2': token_ids2,
            'token_type_ids_2': token_type_ids2,
            'attention_mask_2': attention_mask2,
            'labels': labels,
            'sent_ids': sent_ids,
            'pos_tag_ids_1': pos_tags_ids,
            'ner_tag_ids_1': ner_tags_ids,
            'pos_tag_ids_2': pos_tags_ids2,
            'ner_tag_ids_2': ner_tags_ids2
        }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args, additional_input=False):
        self.dataset = dataset
        self.p = args
        self.additional_input = additional_input

        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")

        pos_tags_spacy = self.nlp.get_pipe("tagger").labels
        ner_tags_spacy = self.nlp.get_pipe("ner").labels

        # Create a vocabulary dictionary for tags
        self.pos_tag_vocab = {tag: index + 1 for index, tag in enumerate(pos_tags_spacy)}
        self.ner_tag_vocab = {tag: index + 1 for index, tag in enumerate(ner_tags_spacy)}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])

        pos_tags_ids, ner_tags_ids = get_pos_ner_tags(self.additional_input, token_ids, self.nlp, self.pos_tag_vocab,
                                                      self.ner_tag_vocab)

        pos_tags_ids2, ner_tags_ids2 = get_pos_ner_tags(self.additional_input, token_ids2, self.nlp, self.pos_tag_vocab,
                                                        self.ner_tag_vocab)

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                sent_ids, pos_tags_ids, ner_tags_ids, pos_tags_ids2, ner_tags_ids2)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         sent_ids, pos_tags_ids, ner_tags_ids, pos_tags_ids2, ner_tags_ids2) = self.pad_data(all_data)

        batched_data = {
            'token_ids_1': token_ids,
            'token_type_ids_1': token_type_ids,
            'attention_mask_1': attention_mask,
            'token_ids_2': token_ids2,
            'token_type_ids_2': token_type_ids2,
            'attention_mask_2': attention_mask2,
            'sent_ids': sent_ids,
            'pos_tag_ids_1': pos_tags_ids,
            'ner_tag_ids_1': ner_tags_ids,
            'pos_tag_ids_2': pos_tags_ids2,
            'ner_tag_ids_2': ner_tags_ids2
        }

        return batched_data


def load_multitask_test_data():
    paraphrase_filename = f'data/quora-test.csv'
    sentiment_filename = f'data/ids-sst-test.txt'
    similarity_filename = f'data/sts-test.csv'

    sentiment_data = []

    with open(sentiment_filename, 'r', encoding='utf-8') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            sent = record['sentence'].lower().strip()
            sentiment_data.append(sent)

    print(f"Loaded {len(sentiment_data)} test examples from {sentiment_filename}")

    paraphrase_data = []
    with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            # if record['split'] != split:
            #    continue
            paraphrase_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(paraphrase_data)} test examples from {paraphrase_filename}")

    similarity_data = []
    with open(similarity_filename, 'r', encoding='utf-8') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            similarity_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(similarity_data)} test examples from {similarity_filename}")

    return sentiment_data, paraphrase_data, similarity_data


def load_multitask_data(sentiment_filename, paraphrase_filename, similarity_filename, split='train'):
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent, sent_id))
    else:
        with open(sentiment_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label, sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(record['sentence2']),
                                            int(float(record['is_duplicate'])), sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        , sent_id))
    else:
        with open(similarity_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']), sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data
