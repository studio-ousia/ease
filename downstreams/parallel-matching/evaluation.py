import argparse
import os
import sys

import numpy as np
import pycountry
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from prettytable import PrettyTable
from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer


sys.path.append(os.path.abspath(os.getcwd()))
from utils.utils import get_mlflow_writer, print_table


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
    if args.pooler == "cls":
        # There is a linear+activation layer after CLS representation
        return pooler_output.cpu()
    elif args.pooler == "cls_before_pooler":
        return last_hidden[:, 0].cpu()
    elif args.pooler == "avg":
        return (
            (last_hidden * batch["attention_mask"].unsqueeze(-1)).sum(1)
            / batch["attention_mask"].sum(-1).unsqueeze(-1)
        ).cpu()
    elif args.pooler == "avg_first_last":
        first_hidden = hidden_states[0]
        last_hidden = hidden_states[-1]
        pooled_result = (
            (first_hidden + last_hidden) / 2.0 * batch["attention_mask"].unsqueeze(-1)
        ).sum(1) / batch["attention_mask"].sum(-1).unsqueeze(-1)
        return pooled_result.cpu()
    elif args.pooler == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        pooled_result = (
            (last_hidden + second_last_hidden)
            / 2.0
            * batch["attention_mask"].unsqueeze(-1)
        ).sum(1) / batch["attention_mask"].sum(-1).unsqueeze(-1)
        return pooled_result.cpu()
    else:
        raise NotImplementedError


# calculate cosine sim matrix
def get_cos_sim_matrix(matrix1, matrix2):
    d = np.dot(matrix1, matrix2.T)
    norm1 = (matrix1 * matrix1).sum(axis=1, keepdims=True) ** 0.5
    norm2 = (matrix2 * matrix2).sum(axis=1, keepdims=True) ** 0.5
    return d / (np.dot(norm1, norm2.T))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Transformers' model name or path",
        default="bert-base-multilingual-cased",
    )
    parser.add_argument(
        "--pooler",
        type=str,
        choices=["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        default="cls_before_pooler",
        help="Which pooler to use",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="eval_tatoeba",
        help="mlflow experiment name",
    )
    parser.add_argument(
        "--langs",
        type=str,
        nargs="+",
        default=[
            "ar",
            "ca",
            "cs",
            "de",
            "eo",
            "es",
            "fa",
            "fr",
            "it",
            "ja",
            "ko",
            "nl",
            "pl",
            "pt",
            "ru",
            "sv",
            "tr",
        ]
        # "--langs", type=str, nargs="+", default=["kab","pam","kw","br","mhr"]
    )

    args = parser.parse_args()
    print("model_path", args.model_name_or_path)
    # mlflow
    mlflow_writer = get_mlflow_writer(args.experiment_name, "mlruns", OmegaConf.create({"eval_args": vars(args)}))

    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path)

    if "xlm" in args.model_name_or_path:
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    lang_2_to_3 = {
        lang.alpha_2: lang.alpha_3
        for lang in pycountry.languages
        if hasattr(lang, "alpha_2")
    }
    lang_3_to_2 = {v: k for k, v in lang_2_to_3.items()}
    langs = [lang_2_to_3[lang] if lang in lang_2_to_3 else lang for lang in args.langs]
    langs = sorted(set(langs), key=langs.index)

    dataset = dict()
    not_exist_langs = set()

    for lang in langs:
        try:
            src_path = f"downstreams/parallel-matching/tatoeba/v1/tatoeba.{lang}-eng.{lang}"
            trg_path = f"downstreams/parallel-matching/tatoeba/v1/tatoeba.{lang}-eng.eng"
            dataset[lang] = (load_data(src_path), load_data(trg_path))
        except:
            print(f"{lang} doesn't exist.")
            not_exist_langs.add(lang)

    langs = sorted(set(langs) - not_exist_langs, key=langs.index)
    scores = []
    for lang in langs:
        print(lang)
        sentence1, sentence2 = dataset[lang]
        
        # TODO batch process

        source_embeddings = batcher(model, tokenizer, sentence1, args, device=device)
        target_embeddings = batcher(model, tokenizer, sentence2, args, device=device)

        lang_to_en_result = (
            get_cos_sim_matrix(source_embeddings, target_embeddings).argmax(axis=1)
            == np.arange(len(source_embeddings))
        ).sum() / 10

        en_to_lang_result = (
            get_cos_sim_matrix(target_embeddings, source_embeddings).argmax(axis=1)
            == np.arange(len(target_embeddings))
        ).sum() / 10

        scores.append("%.2f" % ((lang_to_en_result + en_to_lang_result) / 2))

        if lang in lang_3_to_2:
            mlflow_writer.log_metric(
                lang_3_to_2[lang], (lang_to_en_result + en_to_lang_result) / 2
            )
        else:
            mlflow_writer.log_metric(lang, (lang_to_en_result + en_to_lang_result) / 2)

    langs.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    mlflow_writer.log_metric(
        f"Avg.", (sum([float(score) for score in scores]) / len(scores))
    )
    print_table(langs, scores)


if __name__ == "__main__":
    main()
