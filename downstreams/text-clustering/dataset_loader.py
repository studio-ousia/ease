import ast
import csv
import json
import os
import re
import sys
import unicodedata

from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

WHITESPACE_REGEXP = re.compile(r"\s+")


class MLDocParser:
    def __call__(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                label, sentence = line.strip().split("\t")
                # clean the sentence
                sentence = re.sub(r"\u3000+", "\u3000", sentence)
                sentence = re.sub(r" +", " ", sentence)
                sentence = re.sub(r"\(c\) Reuters Limited \d\d\d\d", "", sentence)
                #                 sentence = normalize_text(sentence)
                yield sentence, label


def normalize_text(text):
    text = text.lower()
    text = re.sub(WHITESPACE_REGEXP, " ", text)
    # remove accents: https://stackoverflow.com/a/518232
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    text = unicodedata.normalize("NFC", text)
    return text


def dataset_load(key):
    
    print(os.getcwd())

    key_to_data_path = {
        "Tweet": "downstreams/text-clustering/data/monolingual_benchmark/tweet.txt",
        "Bio": "downstreams/text-clustering/data/monolingual_benchmark/biomedical.txt",
        "SO": "downstreams/text-clustering/data/monolingual_benchmark/stackoverflow.txt",
        "SS": "downstreams/text-clustering/data/monolingual_benchmark/searchsnippets.txt",
        "AG": "downstreams/text-clustering/data/monolingual_benchmark/agnews.txt",
        "G-TS": "downstreams/text-clustering/data/monolingual_benchmark/ts.txt",
        "G-T": "downstreams/text-clustering/data/monolingual_benchmark/t.txt",
        "G-S": "downstreams/text-clustering/data/monolingual_benchmark/s.txt",
        "WN-FS-ar": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-ca": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-cs": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-de": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-en": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-es": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-eo": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-fa": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-fr": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-ko": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-ja": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-pl": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-pt": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-ru": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-sv": "downstreams/text-clustering/data/mewsc16",
        "WN-FS-tr": "downstreams/text-clustering/data/mewsc16",
    }

    key_to_label_path = {
        "Bio": "downstreams/text-clustering/data/monolingual_benchmark/biomedical_label.txt",
        "SO": "downstreams/text-clustering/data/monolingual_benchmark/stackoverflow_label.txt",
        "SS": "downstreams/text-clustering/data/monolingual_benchmark/searchsnippets_label.txt",
    }

    if key in key_to_data_path:
        data_path = key_to_data_path[key]

    if key in key_to_label_path:
        label_path = key_to_label_path[key]

    lang_pos = []
    data_num = 0

    if key == "Tweet":
        with open(data_path) as f:
            l_strip = [s.strip() for s in f.readlines()]
        sentences = [ast.literal_eval(d)["text"] for d in l_strip]
        labels = [ast.literal_eval(d)["cluster"] for d in l_strip]

    elif key in ["Bio", "SO", "SS"]:
        with open(data_path) as f:
            sentences = [s.strip() for s in f.readlines()]
        with open(label_path) as f:
            labels = [int(s.strip()) for s in f.readlines()]

    elif key in ["AG", "G-TS", "G-T", "G-S"]:
        with open(data_path) as f:
            l_strip = [s.strip() for s in f.readlines()]
        sentences = [d.split("\t")[1] for d in l_strip]
        labels = [int(d.split("\t")[0]) for d in l_strip]

    elif key.startswith("WN"):

        if key.startswith("WN-unified"):
            with open(
                "/home/fmg/nishikawa/EASE/text-clustering/wikinews_clustering/en_cat_to_lang_cat.json"
            ) as f:
                en_cat_to_lang_cat = json.load(f)

            sentences, categories = [], []

            for lang in [
                "en",
                "ar",
                "ja",
                "es",
                "tr",
                "ko",
                "pl",
                "fa",
                "ru",
                "de",
                "fr",
                "ca",
                "cs",
                "eo",
                "pt",
                "sv",
            ]:
                if lang != "en":
                    lang_cat_to_en_cat = {
                        lang_dict[lang]: en_cat
                        for en_cat, lang_dict in en_cat_to_lang_cat.items()
                        if lang in lang_dict
                    }
                else:
                    lang_cat_to_en_cat = {
                        en_cat: en_cat for en_cat in en_cat_to_lang_cat.keys()
                    }

                lang_data_path = data_path + "/" + lang + "_sentences.txt"
                lang_label_path = data_path + "/" + lang + "_categories.txt"
                with open(lang_data_path) as f:
                    tmp_sentences = [s.strip() for s in f.readlines()]

                with open(lang_label_path) as f:
                    tmp_categories = [s.strip() for s in f.readlines()]

                tmp_sentences, tmp_categories = zip(
                    *[
                        (sentence, lang_cat_to_en_cat[category])
                        for sentence, category in zip(tmp_sentences, tmp_categories)
                        if category in lang_cat_to_en_cat
                    ]
                )

                sentences.extend(tmp_sentences)
                categories.extend(tmp_categories)
                lang_pos.append((lang, data_num, data_num + len(tmp_sentences)))
                data_num += len(tmp_sentences)

        else:
            lang = key[-2:]
            label_path = data_path + "/" + lang + "_categories.txt"

            if key.startswith("WN-TFS"):
                data_path = data_path + "/" + lang + "_title_and_sentences.txt"
            elif key.startswith("WN-TS"):
                data_path = data_path + "/" + lang + "_title_and_texts.txt"
            elif key.startswith("WN-T"):
                data_path = data_path + "/" + lang + "_titles.txt"
            elif key.startswith("WN-S"):
                data_path = data_path + "/" + lang + "_texts.txt"
            elif key.startswith("WN-FS"):
                data_path = data_path + "/" + lang + "_sentences.txt"

            with open(data_path) as f:
                sentences = [s.strip() for s in f.readlines()]

            with open(label_path) as f:
                categories = [s.strip() for s in f.readlines()]

            # for sentence tokenizer error
            if key.startswith("WN-FS"):
                sentences, categories = zip(
                    *[
                        (sentence, category)
                        for sentence, category in zip(sentences, categories)
                        if len(sentence) > 0
                    ]
                )
                sentences, categories = list(sentences), list(categories)
        category_to_idx = {
            category: idx for idx, category in enumerate(set(categories))
        }
        labels = [category_to_idx[category] for category in categories]
    return sentences, labels, lang_pos
