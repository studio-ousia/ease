"""
Wikipediaからランダムに抽出された文データセットを構築
"""

import os
import sys
from wikipedia2vec.dump_db import DumpDB
from tqdm import tqdm
from multiprocessing import Process, Manager, Value
from typing import List
import random
import argparse
import gc


sys.path.append(os.path.abspath(".."))
from utils.sentence_tokenizer import MultilingualSentenceTokenizer
from utils.utils import pickle_dump, pickle_load

def filter_size(data, sample_size):
    data_v = random.sample(list(data.values()), min(len(data), sample_size))
    return {i: d for i, d in enumerate(data_v)}

def process(data):
    (entity_per_sentence, sentence_tokenizer, paragraph, args, error_num) = data

    try:
        sentence_list = sentence_tokenizer.tokenize(paragraph.text)
    except Exception as e:
        print("=== sentence tokenize error ===")
        print(e)
        print("sentence:" + paragraph.text)
        error_num.value += 1
        return

    for sentence in sentence_list:
        data_idx = len(entity_per_sentence)
        entity_per_sentence[data_idx] = {
            "positive_entity": "[PAD]",
            "negative_entity": [],
            "text": sentence,
            "origin_entity": "[PAD]",
            "masked_text": sentence,
        }


def main():

    parser = argparse.ArgumentParser() 
    parser.add_argument("output_dir", type=str, default="/home/fmg/nishikawa/EASE/data/")
    parser.add_argument("wikipedia2vec_dump_db_path", type=str, default="/home/fmg/nishikawa/shinra_data/shinra/entity_db/en.db")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    print("load files...")
    random.seed(42)
    wiki_db = DumpDB(args.wikipedia2vec_dump_db_path)
    # db_file_path = f"/home/fmg/nishikawa/shinra_data/shinra/entity_db/{args.lang}.db"
    # if not os.path.isfile(db_file_path):
    #     db_file_path = f"/home/fmg/nishikawa/build_wiki_dbs/dump_dbs/{args.lang}.db"

    print("set tokenizer...")
    sentence_tokenizer = MultilingualSentenceTokenizer(args.lang)

    print("set manager dicts...")
    manager = Manager()
    error_num = Value("i", 0)
    entity_per_sentence = manager.dict()
    paragraphs_with_titles = []

    print("preprocessing...")

    cnt = 0

    for idx, lang_title in tqdm(enumerate(wiki_db.titles())):
        if args.test_mode:
            if idx >= 100:
                break

        try:
            paragraphs = [paragraph for paragraph in wiki_db.get_paragraphs(lang_title)]
            paragraphs_with_titles.extend(
                [(paragraph, lang_title) for paragraph in paragraphs]
            )
            cnt += 1
        except:
            pass

    datas = [
        (entity_per_sentence, sentence_tokenizer, paragraph, args, error_num)
        for paragraph, origin_lang_title in paragraphs_with_titles
    ]
    print(len(datas))

    # p = Pool(16)

    print("processing...")
    for data in tqdm(datas):
        process(data)
    gc.collect()

    entity_per_sentence = dict(entity_per_sentence)
    print("sample 1m")
    entity_per_sentence = filter_size(entity_per_sentence, 1000000)
    print(f"error num: {error_num.value}")

    print("saving...")
    pickle_dump(
        entity_per_sentence,
        os.path.join(args.output_dir, f"wikidata_randam_sentences_{args.lang}.pkl")
    )


if __name__ == "__main__":
    main()
