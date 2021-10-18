from transformers import AutoModel, AutoTokenizer
from coclust.evaluation.external import accuracy
from sklearn.cluster import KMeans 
import torch
import torch.nn as nn
import numpy as np
import argparse
from dataset_loader import dataset_load
from prettytable import PrettyTable
from tqdm import tqdm

import os

os.chdir('/home/fmg/nishikawa/EASE/text-clustering')

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, help="Transformers' model name or path", default="bert-base-uncased"
    )
    parser.add_argument(
        "--pooler",
        type=str,
        choices=["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        # default="cls_before_pooler",
        default="avg",
        help="Which pooler to use",
    )

    args = parser.parse_args()
    print("model_path", args.model_name_or_path)

    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    scores = []
    # datasets = ["tweet", "biomedical", "stackoverflow", "searchsnippets", "agnews", "googlenews_ts", "googlenews_t", "googlenews_s"] 
    datasets = ['AG', 'SS', 'SO', 'Bio', 'Tweet', 'G-TS', 'G-S', 'G-T']
    batch_size = 256

    for dataset_key in datasets:
        print(f"evaluate {dataset_key}...")
        sentences, labels = dataset_load(dataset_key)

        print("encode sentence embeddings...")
        sentence_embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            sentence_embeddings.append(batcher(model, tokenizer, sentences[i: i + batch_size], args, device="cuda"))
        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

        # sentence_embeddings = batcher(model, tokenizer, sentences, args, device="cuda")
        print("clutering...")
        kmeans_model = KMeans(n_clusters=len(set(labels)), random_state=12).fit(sentence_embeddings)
        pred_labels = kmeans_model.labels_
        acc = accuracy(labels, pred_labels) * 100
        print("%.1f" % (acc))
        scores.append("%.1f" % (acc))

    datasets.append("Avg.")
    scores.append("%.1f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(datasets, scores)


if __name__ == "__main__":
    main()
