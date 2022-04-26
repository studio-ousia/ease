import gc
import os
import random
import sys
from abc import ABC, ABCMeta, abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from utils.utils import pickle_load

ENTITY_PAD_MARK = "[PAD]"


def get_dataset(data, max_seq_length, tokenizer, entity_vocab, model_args):

    input_ids = []
    attention_masks = []
    token_type_ids = []
    title_ids = []
    hn_title_ids = []

    for d in tqdm(data):
        title, sentence, hn_titles = (
            d["positive_entity"],
            d["text"],
            d["negative_entity"],
        )

        # TODO how to choose hn title
        # TODO fix bug for multiple hardnegatives
        hn_titles = random.sample(hn_titles, 1)

        sent_features = tokenizer(
            sentence, max_length=max_seq_length, truncation=True, padding="max_length"
        )

        features = {}
        for key in sent_features:
            features[key] = [sent_features[key], sent_features[key]]

        if title in entity_vocab:
            title_ids.append(entity_vocab[title])
        else:
            title_ids.append(entity_vocab[ENTITY_PAD_MARK])

        hn_title_ids.append(
            np.array(
                [
                    entity_vocab[hn_title]
                    if hn_title in entity_vocab
                    else entity_vocab[ENTITY_PAD_MARK]
                    for hn_title in hn_titles
                ],
                dtype=int,
            )
        )
        input_ids.append(features["input_ids"])
        attention_masks.append(features["attention_mask"])
        if "token_type_ids" in features:
            token_type_ids.append(features["token_type_ids"])
    return MyDataset(
        input_ids,
        attention_masks,
        token_type_ids,
        title_ids,
        hn_title_ids,
        model_args.model_name_or_path,
    )


class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        title_id,
        hn_title_ids,
        bert_model,
    ):
        self.bert_model = bert_model
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        if "roberta" not in self.bert_model:
            self.token_type_ids = token_type_ids
        self.title_id = title_id
        self.hn_title_ids = hn_title_ids

    def __getitem__(self, idx):
        item = dict()
        item["input_ids"] = self.input_ids[idx]
        item["attention_mask"] = self.attention_mask[idx]
        if "roberta" not in self.bert_model:
            item["token_type_ids"] = self.token_type_ids[idx]
        item["title_id"] = self.title_id[idx]
        item["hn_title_ids"] = self.hn_title_ids[idx]
        return item

    def __len__(self):
        return len(self.input_ids)


# sizeまでPAD_MARKで埋める
def add_padding(data_list, size):
    if len(data_list) < size:
        data_list = data_list + [ENTITY_PAD_MARK] * (size - len(data_list))
    return data_list
