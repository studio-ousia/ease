from utils.utils import pickle_load
from abc import ABC, ABCMeta, abstractmethod
import sys
import random
import torch
from tqdm import tqdm

# from datasets import load_dataset
import gc
import os
import numpy as np

ENTITY_PAD_MARK = "[PAD]"


def get_dataset(data, max_seq_length, tokenizer, entity_vocab, masked_sentence_ratio):

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
    return input_ids, attention_masks, token_type_ids, title_ids, hn_title_ids


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


# 抽象クラス
# https://qiita.com/baikichiz/items/7c3fdb721bb72644f638
class AbstractDataFormatter(metaclass=ABCMeta):
    def __init__(self, file_obj, sample_num, hard_negative_num, min_length):
        self.file_obj = file_obj
        self.min_length = min_length
        self.sample_num = sample_num
        self.hard_negative_num = hard_negative_num

    @abstractmethod
    def format(self):
        """
        Returns:
          (title, sentence, hn_title)のリスト型?
        """
        pass


# wikipedia abstractから作成したデータのフォーマッタ
class WikiAbstDataFormatter(AbstractDataFormatter):
    def format(self):
        print("formatting...")
        res = []
        for idx, data in tqdm(self.file_obj.items()):
            if len(data["text"]) >= self.min_length:
                res.append(
                    (
                        data["positive_entity"],
                        data["text"],
                        random.sample(data["negative_entity"], self.hard_negative_num),
                    )
                )
        return random.sample(res, min(len(res), self.sample_num))


# wikidataから作成したデータのフォーマッタ
class WikidataDataFormatter(AbstractDataFormatter):
    def __init__(
        self,
        file_obj,
        sample_num,
        hard_negative_num,
        min_length,
    ):
        super().__init__(
            file_obj, sample_num, hard_negative_num, min_length
        )  # 親クラスの初期化メソッドを呼び出す

    def format(self):
        print("formatting...")
        res = []
        for file_obj in self.file_obj:
            lang_res = []
            for idx, data in tqdm(file_obj.items()):

                if len(data["text"]) >= self.min_length:
                    masked_sentence = None
                    if "masked_text" in data:
                        masked_sentence = data["masked_text"]
                    lang_res.append(
                        (
                            data["positive_entity"],
                            data["text"],
                            random.sample(
                                add_padding(
                                    data["negative_entity"], self.hard_negative_num
                                ),
                                self.hard_negative_num,
                            ),
                            masked_sentence,
                        )
                    )
            res.extend(random.sample(lang_res, min(len(lang_res), self.sample_num)))
        return res


# simCSEのオリジナルのデータのフォーマッタ
class SimCSEDataFormatter(AbstractDataFormatter):
    def format(self):
        print("formatting...")
        res = []
        for data in tqdm(self.file_obj["train"]):
            # ダミーエンティティ
            res.append(
                (
                    ENTITY_PAD_MARK,
                    data["text"],
                    [ENTITY_PAD_MARK] * self.hard_negative_num,
                )
            )
        res = random.sample(res, min(len(res), self.sample_num))

        return res


# データの形式に合わせてデータセットを作成するクラス
# いろんな形式 -> (title, sentence, hn_title)に変換
class RawDataLoader:
    @staticmethod
    def load(
        cwd,
        dataset,
        sample_num=100000,
        hard_negative_num=1,
        min_length=5,
        langs=["en", "ar", "es", "tr"],
        seed=42,
    ):
        random.seed(seed)

        # # ここにloaderを追加していく
        # if dataset in ["wiki_hyperlink", "wiki_first-sentence"]:
        #     file_obj = pickle_load(dataset_path)
        #     formatter = WikiAbstDataFormatter(
        #         file_obj, sample_num, hard_negative_num, min_length
        #     )
        if dataset.startswith("wikidata_hyperlink_type_hn"):
            print(f"langs: {langs}")
            file_objs = []
            for lang in langs:

                if lang in ["en", "ar", "es", "tr"]:
                    dataset_path = f"data/wikidata_hyperlinks_with_type_hardnegatives_abst_False_1m_{lang}.pkl"

                else:
                    dataset_path = f"data/wikidata_hyperlinks_with_type_hardnegatives_test_False_first_sentence_False_abst_False_size_1000000_max_count_10000_link_min_count_10_{lang}.pkl"

                # elif lang in ['fr','de','ja','zh','it','ru','nl']:
                #     dataset_path = f"data/wikidata_hyperlinks_with_type_hardnegatives_test_False_first_sentence_False_abst_False_size_1000000_max_count_10000_link_min_count_10_{lang}.pkl"

                # else:
                #     dataset_path = f"data/wikidata_hyperlinks_with_type_hardnegatives_test_False_first_sentence_False_abst_False_{lang}.pkl"

                if dataset == "wikidata_hyperlink_type_hn_paragraph":
                    dataset_path = f"data/wikidata_hyperlinks_with_type_hardnegatives_test_False_first_sentence_False_abst_False_paragraph_True_size_1000000_max_count_10000_link_min_count_10_{lang}.pkl"

                if dataset == "wikidata_hyperlink_type_hn_high_freq":
                    dataset_path = f"data/wikidata_hyperlinks_with_type_hardnegatives_test_False_first_sentence_False_abst_False_size_1000000_max_count_10000_link_min_count_50_{lang}.pkl"

                dataset_path = os.path.join(cwd, dataset_path)

                try:
                    file_objs.append(pickle_load(dataset_path))

                except FileNotFoundError:
                    print(f"{dataset_path} file not found error!")
                    return

            formatter = WikidataDataFormatter(
                file_objs,
                sample_num,
                hard_negative_num,
                min_length,
            )

        elif dataset == "wikipedia_random":
            print(f"langs: {langs}")
            file_objs = []
            for lang in langs:
                dataset_path = f"data/wikidata_randam_sentences_{lang}.pkl"
                dataset_path = os.path.join(cwd, dataset_path)
                file_objs.append(pickle_load(dataset_path))
            formatter = WikidataDataFormatter(
                file_objs,
                sample_num,
                hard_negative_num,
                min_length,
            )

        # elif dataset.startswith("wikidata_hyperlink"):
        #     print(f"langs: {langs}")
        #     file_objs = [pickle_load(f"{dataset_path}_{lang}.pkl") for lang in langs]
        #     formatter = WikidataDataFormatter(
        #         file_objs,
        #         sample_num,
        #         hard_negative_num,
        #         min_length,
        #     )
        # elif dataset == "SimCSE_original":
        #     file_obj = load_dataset(
        #         "text", data_files=dataset_path, cache_dir="./data/"
        #     )
        #     formatter = SimCSEDataFormatter(file_obj,  sample_num, hard_negative_num, 0)
        else:
            print("Error: 該当のモードがありません", file=sys.stderr)
            sys.exit(1)
        return formatter.format()
