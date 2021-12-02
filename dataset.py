from utils.utils import pickle_load
from abc import ABC, ABCMeta, abstractmethod
import sys
import random
from tqdm import tqdm
from datasets import load_dataset
import gc
import os

ENTITY_PAD_MARK = "[PAD]"

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
                            masked_sentence
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
                    print(f'{dataset_path} file not found error!')

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
