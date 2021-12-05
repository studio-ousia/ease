from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import pycountry
from omegaconf import OmegaConf
import argparse
import csv
from collections import defaultdict
from tqdm import tqdm

from torch.nn.functional import cosine_similarity as cossim
from collections import Counter

sys.path.append(os.path.abspath(".."))
from utils.mlflow_writer import MlflowWriter


def data_load(path):
    data = defaultdict(list)
    langs = ["de", "en", "es", "fr", "ja", "ko", "zh"]
    all_read_data = []
    for lang in langs:
        # if lang == "en":
        #     lang_path = os.path.join(path, f"{lang}/train.tsv")
        # else:
        #     lang_path = os.path.join(path, f"{lang}/translated_train.tsv")
        lang_path = os.path.join(path, f"{lang}/test_2k.tsv")
        with open(lang_path, mode="r", newline="", encoding="utf-8") as f:
            tsv_reader = csv.reader(f, delimiter="\t")
            read_data = [row for row in tsv_reader]
        all_read_data.append(read_data)
    all_read_data = list(map(list, zip(*all_read_data)))
    for datas in all_read_data[1:]:
        if datas[0][-1] == "1" and all([len(d) == 4 for d in datas]):
            for lang, d in zip(langs, datas):
                data[lang].append([d[1], d[2]])
    return data


def batcher(model, tokenizer, sentence, args, device="cuda"):
    # encode_dict = tokenizer.batch_encode_plus(sentence, pad_to_max_length=True, add_special_tokens=True, return_tensors="pt",)
    batch = tokenizer.batch_encode_plus(
        sentence, return_tensors="pt", padding=True, max_length=512, truncation=True
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Transformers' model name or path",
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--pooler",
        type=str,
        choices=["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        default="avg",
        help="Which pooler to use",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="eval_pawsx",
        help="mlflow experiment name",
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default="de",
    )

    args = parser.parse_args()
    print("model_path", args.model_name_or_path)
    # mlflow
    cfg = OmegaConf.create({"eval_args": vars(args)})
    EXPERIMENT_NAME = args.experiment_name
    tracking_uri = f"/home/fmg/nishikawa/EASE/mlruns"
    mlflow_writer = MlflowWriter(EXPERIMENT_NAME, tracking_uri=tracking_uri)
    mlflow_writer.log_params_from_omegaconf_dict(cfg)

    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path)

    if "xlm" in args.model_name_or_path:
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    langs = ["de", "en", "es", "fr", "ja", "ko", "zh"]
    data = data_load("/home/fmg/nishikawa/corpus/x-final/")
    batch_size = 128
    lang_count = Counter()

    for i in tqdm(range(0, len(data[args.source_lang]), batch_size)):
        query = [d[0] for d in data[args.source_lang][i : i + batch_size]]
        sentences = [[d[1] for d in data[lang][i : i + batch_size]] for lang in langs]

        query_embeddings = batcher(
            model, tokenizer, query, args, device="cuda"
        ).unsqueeze(1)
        all_target_embeddings = []
        for idx in range(len(langs)):
            target_embeddings = batcher(
                model, tokenizer, sentences[idx], args, device="cuda"
            ).unsqueeze(1)
            all_target_embeddings.append(target_embeddings)
        all_target_embeddings = torch.cat(all_target_embeddings, dim=1)
        lang_count.update(
            cossim(query_embeddings, all_target_embeddings, dim=2).argmax(1).tolist()
        )

    all_sum = np.sum(list(lang_count.values()))
    for idx, lang in enumerate(langs):
        result = (lang_count[idx] / all_sum) * 100
        print(lang, result)
        mlflow_writer.log_metric(lang, result)


if __name__ == "__main__":
    main()
