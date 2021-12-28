# エンティティの登場回数・ハードネガティブを考慮してサンプ
from tqdm import tqdm
import random
import argparse
from collections import Counter, defaultdict

random.seed(42)

import sys
import os

sys.path.append(os.path.abspath(".."))
from utils.utils import pickle_dump, pickle_load


def filter_hardnegative(data):
    new_data = dict()
    for idx, d in tqdm(data.items()):
        if len(d["negative_entity"]) > 0:
            new_data[idx] = d
    return new_data


def filter_entity_max_count(data, max_count):
    new_data = dict()
    entity_counter = Counter()

    # dataをランダムソート
    print("random sort...")
    data = [(idx, d) for idx, d in data.items()]
    data = random.sample(data, len(data))

    # max_count以下であることを担保しながら代入
    print("substitution..")
    for idx, d in tqdm(data):
        if entity_counter[d["positive_entity"]] < max_count:
            new_data[idx] = d
        entity_counter.update([d["positive_entity"]])
    return new_data


def filter_size(data, sample_size):
    data_v = random.sample(list(data.values()), min(len(data), sample_size))
    return {i: d for i, d in enumerate(data_v)}


def filter_entity_link_min_count(data, min_count):
    new_data = dict()
    en_title_to_link_count = pickle_load(
        "/home/fmg/nishikawa/EASE/data/en_title_to_link_count.pkl"
    )
    for idx, d in tqdm(data.items()):
        if (
            d["positive_entity"] in en_title_to_link_count
            and en_title_to_link_count[d["positive_entity"]] >= min_count
        ):
            new_data[idx] = d
    return new_data


def main():
    parser = argparse.ArgumentParser(description="このプログラムの説明（なくてもよい）")  # 2. パーサを作る
    parser.add_argument("lang", type=str, default="test")
    parser.add_argument("data_path", type=str)
    parser.add_argument("--sample_size", type=int, default=1000000)
    parser.add_argument("--max_count", type=int, default=10000)
    parser.add_argument("--link_min_count", type=int, default=10)
    args = parser.parse_args()

    data = pickle_load(args.data_path)

    data = filter_hardnegative(data)
    data = filter_entity_max_count(data, args.max_count)
    data = filter_entity_link_min_count(data, args.link_min_count)
    data = filter_size(data, args.sample_size)

    before_pkl_idx = args.data_path.index(".pkl") - (1 + len(args.lang))
    lang = args.lang
    
    output_path = (
        args.data_path[:before_pkl_idx]
        + "_size_"
        + str(args.sample_size)
        + "_max_count_"
        + str(args.max_count)
        + "_link_min_count_"
        + str(args.link_min_count)
        + "_"
        + lang
        + ".pkl"
    )

    pickle_dump(data, output_path)


if __name__ == "__main__":
    main()
