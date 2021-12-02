from sklearn.utils import shuffle
import pickle
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import numpy as np
import copy
import os
import shutil
import importlib
import random
import logging
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import torch
from xml.etree import ElementTree
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

PAD_TOKEN = "<PAD>"
WHITESPACE_REGEXP = re.compile(r"\s+")


class DatasetInstance(object):
    def __init__(self, text, label, fold, title=None, page_id=None):
        self.text = text
        self.label = label
        self.fold = fold
        # add
        self.title = title
        self.page_id = page_id


class Dataset(object):
    def __init__(self, name, instances, label_names):
        self.name = name
        self.instances = instances
        self.label_names = label_names

    def __iter__(self):
        for instance in self.instances:
            yield instance

    def __len__(self):
        return len(self.instances)

    def get_instances(self, fold=None):
        if fold is None:
            return self.instances
        else:
            return [ins for ins in self.instances if ins.fold == fold]


def generate_features(dataset, tokenizer, min_count, max_word_length):
    def create_numpy_sequence(source_sequence, length, dtype):
        ret = np.zeros(length, dtype=dtype)
        source_sequence = source_sequence[:length]
        ret[: len(source_sequence)] = source_sequence
        return ret

    #     logger.info('Creating vocabulary...')
    # ここいらない
    word_counter = Counter()
    for instance in tqdm(dataset):
        sentence = instance.text.lower()
        tokenized = tokenizer.tokenize(sentence, return_str=True)
        tokenized_list = tokenized.split()
        word_counter.update(token for token in tokenized_list)

    words = [word for word, count in word_counter.items() if count >= min_count]
    word_vocab = {word: index for index, word in enumerate(words, 1)}
    word_vocab[PAD_TOKEN] = 0

    ret = dict(train=[], dev=[], test=[], word_vocab=word_vocab)

    for fold in ("train", "dev", "test"):
        for instance in dataset.get_instances(fold):
            sentence = instance.text.lower()
            tokenized = tokenizer.tokenize(sentence, return_str=True)
            tokenized_list = tokenized.split()
            if len(tokenized_list) < 3:
                continue
            word_ids = [
                word_vocab[token] for token in tokenized_list if token in word_vocab
            ]
            ret[fold].append(
                dict(
                    word_ids=create_numpy_sequence(word_ids, max_word_length, np.int64),
                    label=instance.label,
                )
            )

    return ret


def load_amazon_dataset(dataset_path, lang, domain, dev_size=0.05, seed=42):
    # data = {}
    instances = []
    porns = ["negative", "positive"]

    def read(mode, lang, instances):
        x, y = [], []
        for i, porn in enumerate(porns):
            with open(
                f"{dataset_path}/{lang}/{domain}/{mode}/{porn}.txt",
                "r",
                encoding="utf-8",
            ) as f:
                for line in f:
                    if line[-1] == "\n":
                        line = line[:-1]
                    x.append(line)
                    y.append(i)
        instances += [DatasetInstance(text, label, mode) for (text, label) in zip(x, y)]

    read("train", lang, instances)
    read("test", lang, instances)
    read("valid", lang, instances)
    return Dataset("amazon", instances, porns)


class MLDocParser:
    def __call__(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                label, sentence = line.strip().split("\t")
                # clean the sentence
                sentence = re.sub(r"\u3000+", "\u3000", sentence)
                sentence = re.sub(r" +", " ", sentence)
                sentence = re.sub(r"\(c\) Reuters Limited \d\d\d\d", "", sentence)
                sentence = normalize_text(sentence)
                yield sentence, label


class AmazonReviewParser:
    def __call__(self, file_path: str):
        for item in ElementTree.parse(file_path).getroot():

            rating = int(float(item.findtext("rating")))
            if rating == 1 or rating == 2:
                label = "negative"
            elif rating == 4 or rating == 5:
                label = "positive"
            else:
                raise Exception(f"Invalid rating: {rating}")

            sentence = item.findtext("text").strip().replace("\n", " ")
            sentence = normalize_text(sentence)

            yield sentence, label


def set_amazon_dataset(dataset_path, output_path):
    def write_data(file_path, data):
        f = open(file_path, "w")
        for x in data:
            f.write(str(x) + "\n")
        f.close()

    data = {}
    instances = []
    lcode_to_lang = {
        "en": "english",
        "fr": "french",
        "de": "german",
        "ja": "japanese",
        "zh": "chinese",
        "it": "italian",
        "ru": "russian",
        "es": "spanish",
    }
    lang = lcode_to_lang[lang]
    modes = ["test.review", "train.review"]
    categories = ["negative", "positive"]
    domains = ["books", "dvd", "music"]
    categories_index = {t: i for i, t in enumerate(categories)}
    parser = AmazonReviewParser()
    for mode in modes:
        file_path = f"{dataset_path}/{lang}/{domain}/{mode}"
        positive = []
        negative = []
        for sentence, label in parser(file_path):
            if label == "negative":
                negative.append(sentence)
            else:
                positive.append(sentence)
        positive = shuffle_samples(positive)
        negative = shuffle_samples(negative)
        if mode == "test.review":
            file_path_p = f"{output_path}/{lang}/{domain}/test/positive.txt"
            write_data(file_path_p, positive)
            file_path_n = f"{output_path}/{lang}/{domain}/test/negative.txt"
            write_data(file_path_n, negative)
        else:
            positive = shuffle_samples(positive)
            negative = shuffle_samples(negative)
            train_positive, train_negative = positive[:1000], negative[:1000]
            valid_positive, valid_negative = positive[1000:2000], negative[1000:2000]
            tmp = ["train", "valid"]
            file_path_p = f"{output_path}/{lang}/{domain}/train/positive.txt"
            write_data(file_path_p, train_positive)
            file_path_n = f"{output_path}/{lang}/{domain}/train/negative.txt"
            write_data(file_path_n, train_negative)
            file_path_p = f"{output_path}/{lang}/{domain}/valid/positive.txt"
            write_data(file_path_p, valid_positive)
            file_path_n = f"{output_path}/{lang}/{domain}/valid/negative.txt"
            write_data(file_path_n, valid_negative)


# def get_labels(ENE_ids, ENE_id_index):
#     labels = [0]*len(ENE_id_index)
#     for d in ENE_ids:
#         labels[ENE_id_index[d['ENE_id']]] = 1
#     return labels


def get_labels(ENE_ids, ENE_id_index):
    labels = []
    for d in ENE_ids:
        labels.append(ENE_id_index[d["ENE_id"]])
    return labels


def load_shinra_dataset(
    shinra_data_path, ENE_id_index, lang, use_singlelabel, use_split_data, seed=42
):
    ene_list_path = (
        f"{shinra_data_path}/preprocess/minimum_{lang}/{lang}/{lang}_ENEW_LIST.json"
    )
    dataset_path = f"{shinra_data_path}/processed/HAND+AUTO/{lang}.json"

    if lang == "ja":
        hand_data = [
            json.loads(l)
            for l in open(
                f"{shinra_data_path}/preprocess/ENEW_ENEtag_20191023.json",
                encoding="utf-8",
            )
        ]
        title_enes = dict()
        for d in hand_data:
            if "HAND.AIP.201910" in d["ENEs"]:
                title_enes[d["title"]] = d["ENEs"]["HAND.AIP.201910"]

            else:
                title_enes[d["title"]] = d["ENEs"]["AUTO.TOHOKU.201906"]

    else:
        hand_data = [json.loads(l) for l in open(ene_list_path, encoding="utf-8")]
        title_enes = {d["title"]: d["ENEs"] for d in hand_data}

    with open(dataset_path, "r", encoding="utf-8") as f:
        load_data = json.load(f)

    instances = []
    x, y, z = [], [], []
    for title, text in tqdm(load_data.items()):
        try:
            ENE_ids = title_enes[title]
            if use_singlelabel:
                # 正解ラベルを一個だけサンプリングする
                ind = random.randint(0, len(ENE_ids) - 1)
                eneid = ENE_ids[ind]["ENE_id"]
                labels = ENE_id_index[eneid]
            else:
                labels = get_labels(ENE_ids, ENE_id_index)
            x.append(text)
            y.append(labels)
            z.append(title)
        except:
            continue
    # for test
    # x, y, z = x[:50000], y[:50000], z[:50000]
    if use_split_data:
        # x_large, x_small, y_train, y_valid, z_train, z_valid  = train_test_split(x,y,z, test_size=0.1, random_state=seed)
        # x_train, x_valid, y_train, y_valid, z_train, z_valid  = train_test_split(x,y,z, test_size=0.05, stratify=y)
        x_train, x_valid, y_train, y_valid, z_train, z_valid = train_test_split(
            x, y, z, test_size=0.05, random_state=seed
        )
        y_train = MultiLabelBinarizer(list(ENE_id_index.values())).fit_transform(
            y_train
        )
        y_valid = MultiLabelBinarizer(list(ENE_id_index.values())).fit_transform(
            y_valid
        )
        instances += [
            DatasetInstance(text, label, "train", title)
            for (text, label, title) in tqdm(zip(x_train, y_train, z_train))
        ]
        instances += [
            DatasetInstance(text, label, "dev", title)
            for (text, label, title) in tqdm(zip(x_valid, y_valid, z_valid))
        ]
    else:
        y = MultiLabelBinarizer(list(ENE_id_index.values())).fit_transform(y)
        instances += [
            DatasetInstance(text, label, "test", title)
            for (text, label, title) in tqdm(zip(x, y, z))
        ]
    return Dataset("shinra", instances, list(ENE_id_index.keys()))


def load_ecls_pretrain_dataset(shinra_data_path, seed=42):
    ene_list_path = (
        f"{shinra_data_path}/preprocess/minimum_{lang}/{lang}/{lang}_ENEW_LIST.json"
    )
    dataset_path = f"{shinra_data_path}/processed/HAND+AUTO/{lang}.json"

    if lang == "ja":
        hand_data = [
            json.loads(l)
            for l in open(
                f"{shinra_data_path}/preprocess/ENEW_ENEtag_20191023.json",
                encoding="utf-8",
            )
        ]
        title_enes = dict()
        for d in hand_data:
            if "HAND.AIP.201910" in d["ENEs"]:
                title_enes[d["title"]] = d["ENEs"]["HAND.AIP.201910"]

            else:
                title_enes[d["title"]] = d["ENEs"]["AUTO.TOHOKU.201906"]

    else:
        hand_data = [json.loads(l) for l in open(ene_list_path, encoding="utf-8")]
        title_enes = {d["title"]: d["ENEs"] for d in hand_data}

    with open(dataset_path, "r", encoding="utf-8") as f:
        load_data = json.load(f)

    instances = []
    x, y, z = [], [], []
    for title, text in tqdm(load_data.items()):
        try:
            x.append(text)
            z.append(title)
        except:
            continue
    x, z = shuffle_samples(x, z)
    x, z = x[:50000], z[:50000]
    x_train, x_valid, z_train, z_valid = train_test_split(
        x, z, test_size=0.05, random_state=seed
    )
    instances += [
        DatasetInstance(text, None, "train", title)
        for (text, title) in tqdm(zip(x_train, z_train))
    ]
    instances += [
        DatasetInstance(text, None, "dev", title)
        for (text, title) in tqdm(zip(x_valid, z_valid))
    ]
    return Dataset("pretrain", instances, None)


def load_mldoc_dataset(dataset_path, lang, dev_size=0.05, seed=1):
    data = {}
    instances = []
    lcode_to_lang = {
        "en": "english",
        "fr": "french",
        "de": "german",
        "ja": "japanese",
        "zh": "chinese",
        "it": "italian",
        "ru": "russian",
        "es": "spanish",
    }
    lang = lcode_to_lang[lang]
    modes = ["train.1000", "dev", "test"]
    categories = ["CCAT", "MCAT", "ECAT", "GCAT"]
    categories_index = {t: i for i, t in enumerate(categories)}
    parser = MLDocParser()
    for mode in modes:
        file_path = f"{dataset_path}/{lang}.{mode}"
        instances += [
            DatasetInstance(sentence, categories_index[label], mode)
            for sentence, label in parser(file_path)
        ]
    return Dataset("mldoc", instances, categories)


# 複数配列に対応
def shuffle_samples(seed=1, *args):
    np.random.seed(seed)
    zipped = list(zip(*args))
    np.random.shuffle(zipped)
    shuffled = list(zip(*zipped))
    result = []
    for ar in shuffled:
        result.append(ar)
    #         result.append(np.asarray(ar))
    return result


def load_ted_dataset(data_path, lang, seed=42):

    dirs = [
        "ar-en",
        "en-ar",
        "en-es",
        "en-it",
        "en-pb",
        "en-ro",
        "en-tr",
        "es-en",
        "it-en",
        "pb-en",
        "ro-en",
        "tr-en",
        "de-en",
        "en-de",
        "en-fr",
        "en-nl",
        "en-pl",
        "en-ru",
        "en-zh",
        "fr-en",
        "nl-en",
        "pl-en",
        "ru-en",
        "zh-en",
    ]
    modes = ["test", "train"]
    # pns = ["negative", "positive"]
    pns = ["positive"]
    categories = [
        "art",
        "arts",
        "biology",
        "business",
        "creativity",
        "culture",
        "design",
        "economics",
        "education",
        "entertainment",
        "global",
        "health",
        "politics",
        "science",
        "technology",
    ]
    category_to_idx = {ca: i for i, ca in enumerate(categories)}
    instances = []

    def get_data(mode, instances):
        all_data = defaultdict(defaultdict)

        for di in tqdm(dirs):
            if lang == di[:2]:
                for category in categories:
                    for pn in pns:
                        base_path = f"{data_path}/{di}/{mode}/{category}/{pn}"
                        files = os.listdir(base_path)
                        for file in files:
                            if file.endswith("ted.txt"):
                                file_path = base_path + "/" + file
                                if file not in all_data:
                                    all_data[file]["text"] = open(file_path, "r").read()
                                    all_data[file]["label"] = set()
                                if pn == "positive":
                                    all_data[file]["label"].add(
                                        category_to_idx[category]
                                    )

        x, y, z = [], [], []
        for key, data in all_data.items():
            sent = data["text"]
            sent = re.sub("\n", " ", sent)
            x.append(sent)
            y.append(list(data["label"]))
            z.append(key)

        if mode == "train":
            x_train, x_valid, y_train, y_valid, z_train, z_valid = train_test_split(
                x, y, z, test_size=0.1, random_state=seed
            )
            y_train = MultiLabelBinarizer(list(category_to_idx.values())).fit_transform(
                y_train
            )
            y_valid = MultiLabelBinarizer(list(category_to_idx.values())).fit_transform(
                y_valid
            )

            instances += [
                DatasetInstance(text, label, "train", title)
                for (text, label, title) in tqdm(zip(x_train, y_train, z_train))
            ]
            instances += [
                DatasetInstance(text, label, "dev", title)
                for (text, label, title) in tqdm(zip(x_valid, y_valid, z_valid))
            ]

        else:
            y = MultiLabelBinarizer(list(category_to_idx.values())).fit_transform(y)
            instances += [
                DatasetInstance(text, label, "test", title)
                for (text, label, title) in tqdm(zip(x, y, z))
            ]

    get_data("train", instances)
    get_data("test", instances)

    return Dataset("ted", instances, list(category_to_idx.keys()))


def normalize_text(text):
    text = text.lower()
    text = re.sub(WHITESPACE_REGEXP, " ", text)
    # remove accents: https://stackoverflow.com/a/518232
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    text = unicodedata.normalize("NFC", text)

    return text
