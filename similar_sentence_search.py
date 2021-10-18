# Tatoebaで評価を行うコード
# コサイン類似度が最も高い文が対訳文かどうか

# import hydra
# from omegaconf import DictConfig, OmegaConf
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
import os
from utils.mlflow_writer import MlflowWriter
import pycountry
import argparse

from evaluation import print_table

# tatoebaのセンテンスをロードする関数
def load_data(path):
    with open(path) as f:
        l_strip = [s.strip() for s in f.readlines()]
        return l_strip


def batcher(model, tokenizer, sentence, args, device="cuda"):
    # encode_dict = tokenizer.batch_encode_plus(sentence, pad_to_max_length=True, add_special_tokens=True, return_tensors="pt",)
    batch = tokenizer.batch_encode_plus(
        sentence,
        return_tensors="pt",
        padding=True,
    )
    # Move to the correct device
    for k in batch:
        batch[k] = batch[k].to(device)
    # Get raw embeddings
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

    # Apply different poolers
    if args.pooler == 'cls':
        # There is a linear+activation layer after CLS representation
        return pooler_output.cpu()
    elif args.pooler == 'cls_before_pooler':
        return last_hidden[:, 0].cpu()
    elif args.pooler == "avg":
        return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
    elif args.pooler == "avg_first_last":
        first_hidden = hidden_states[0]
        last_hidden = hidden_states[-1]
        pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
        return pooled_result.cpu()
    elif args.pooler == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
        return pooled_result.cpu()
    else:
        raise NotImplementedError


# 二つの行列のコサイン類似度行列を作成
def get_cos_sim_matrix(matrix1, matrix2):
    """
    item-feature 行列が与えられた際に
    item 間コサイン類似度行列を求める関数
    """
    d = np.dot(matrix1, matrix2.T)  # item-vector 同士の内積を要素とする行列

    # コサイン類似度の分母に入れるための、各 item-vector の大きさの平方根
    norm1 = (matrix1 * matrix1).sum(axis=1, keepdims=True) ** 0.5
    norm2 = (matrix2 * matrix2).sum(axis=1, keepdims=True) ** 0.5

    # それぞれの item の大きさの平方根で割っている（なんだかスマート！）
    return d / (np.dot(norm1, norm2.T))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, help="Transformers' model name or path", default="bert-base-uncased"
    )
    parser.add_argument(
        "--pooler",
        type=str,
        choices=["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        default="cls_before_pooler",
        help="Which pooler to use",
    )
    parser.add_argument(
        "--langs", type=str, nargs="+", default=["es", "ar", "tr"] 
        # "--langs", type=str, nargs="+", default=['be', 'ga', 'hy', 'kk', 'oc']
        # "--langs", type=str, nargs="+", default=['mhr', 'br', 'kw', 'kzj', 'pam', 'dtp', 'ber', 'kab']
    )

    args = parser.parse_args()
    print("model_path", args.model_name_or_path)

    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    lang_2_to_3 = {
        lang.alpha_2: lang.alpha_3
        for lang in pycountry.languages
        if hasattr(lang, "alpha_2")
    }
    langs = [lang_2_to_3[lang] if lang in lang_2_to_3 else lang for lang in args.langs]

    # 言語ごとにデータを取得
    dataset = dict()

    for lang in langs:
        src_path = f"data/tatoeba/v1/tatoeba.{lang}-eng.{lang}"
        trg_path = f"data/tatoeba/v1/tatoeba.{lang}-eng.eng"
        dataset[lang] = (load_data(src_path), load_data(trg_path))

    lang_to_en_scores = []
    en_to_langs_scores = []

    for lang in langs:
        print(lang)
        sentence1, sentence2 = dataset[lang]

        source_embeddings = batcher(model, tokenizer, sentence1, args, device="cuda")
        target_embeddings = batcher(model, tokenizer, sentence2, args, device="cuda")

        result = (
            get_cos_sim_matrix(source_embeddings, target_embeddings).argmax(axis=1)
            == np.arange(len(source_embeddings))
        ).sum() / 10
        lang_to_en_scores.append("%.1f" % result)
        print(result)

        result = (
            get_cos_sim_matrix(target_embeddings, source_embeddings).argmax(axis=1)
            == np.arange(len(target_embeddings))
        ).sum() / 10
        en_to_langs_scores.append("%.1f" % result)
        print(result)

    langs.append("Avg.")

    print("------ %s ------" % ("lang_to_en"))
    lang_to_en_scores.append("%.1f" % (sum([float(score) for score in lang_to_en_scores]) / len(lang_to_en_scores)))
    print_table(langs, lang_to_en_scores)

    print("------ %s ------" % ("en_to_lang"))
    en_to_langs_scores.append("%.1f" % (sum([float(score) for score in en_to_langs_scores]) / len(en_to_langs_scores)))
    print_table(langs, en_to_langs_scores)


if __name__ == "__main__":
    main()
