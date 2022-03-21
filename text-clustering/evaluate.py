def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer
from coclust.evaluation.external import accuracy
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import numpy as np
import argparse
from dataset_loader import dataset_load
from prettytable import PrettyTable
from tqdm import tqdm
from omegaconf import OmegaConf
from utils.mlflow_writer import MlflowWriter
from utils.glove import GloveSentenceEncoder

from sklearn.metrics.cluster import contingency_matrix
from munkres import Munkres


def get_label_mapping_dict(list1, list2):
    mapping_dict = dict()
    m = Munkres()
    contmat = contingency_matrix(list1, list2)
    idx_pairs = m.compute(contmat.max() - contmat)
    classes, class_idx = np.unique(list1, return_inverse=True)
    clusters, cluster_idx = np.unique(list2, return_inverse=True)
    for l1_idx, l2_idx in idx_pairs:
        mapping_dict[classes[l1_idx]] = clusters[l2_idx]
    return mapping_dict

import os
os.chdir("text-clustering")

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
        truncation=True,
        max_length=512,
    )
    # Move to the correct device
    for k in batch:
        batch[k] = batch[k].to(device)
    # Get raw embeddings
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        if hasattr(outputs, "pooler_output"):
            pooler_output = outputs.pooler_output

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
        default="cls_before_pooler",
        #         default="avg",
        help="Which pooler to use",
    )

    parser.add_argument(
        "--task_set",
        type=str,
        choices=["cl", "mono", "full"],
        default="mono",
        help="Which task",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="evals",
        help="mlflow experiment name",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="seed for Kmeans",
    )

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    print("model_path", args.model_name_or_path)

    # mlflow
    cfg = OmegaConf.create({"eval_args": vars(args)})
    EXPERIMENT_NAME = args.experiment_name
    tracking_uri = f"/home/fmg/nishikawa/EASE/mlruns"
    mlflow_writer = MlflowWriter(EXPERIMENT_NAME, tracking_uri=tracking_uri)
    mlflow_writer.log_params_from_omegaconf_dict(cfg)

    # Load transformers' model checkpoint
    if args.model_name_or_path != "glove":
        model = AutoModel.from_pretrained(args.model_name_or_path)
        if "xlm" in args.model_name_or_path:
            tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    dataset_groups = []
    if args.task_set in ["full", "mono"]:
        # dataset_groups.append(["R8", "R52", "OH", "20N"])
        dataset_groups.append(["AG", "SS", "SO", "Bio", "Tweet", "G-TS", "G-S", "G-T"])

    if args.task_set in ["full", "cl"]:

        # 15 langs label-unified
        dataset_groups.append(
            [
                "WN-FS-ar",
                "WN-FS-ca",
                "WN-FS-cs",
                "WN-FS-de",
                "WN-FS-en",
                "WN-FS-eo",
                "WN-FS-es",
                "WN-FS-fa",
                "WN-FS-fr",
                "WN-FS-ko",
                "WN-FS-ja",
                "WN-FS-pl",
                "WN-FS-pt",
                "WN-FS-ru",
                "WN-FS-sv",
                "WN-FS-tr",
            ]
        )
    batch_size = 256

    results = []

    for datasets in dataset_groups:
        scores = []
        for dataset_key in tqdm(datasets):
            if args.verbose:
                print(f"evaluate {dataset_key}...")
            sentences, labels, lang_pos = dataset_load(dataset_key)

            if args.verbose:
                print("encode sentence embeddings...")
            sentence_embeddings = []

            if args.model_name_or_path == "glove":

                glove_sentence_encoder = GloveSentenceEncoder(
                    "/home/fmg/nishikawa/corpus/glove/glove.6B.300d.txt"
                )
                sentence_embeddings = torch.tensor(
                    glove_sentence_encoder.encode_sentences(sentences)
                )

            else:
                for i in tqdm(range(0, len(sentences), batch_size)):
                    sentence_embeddings.append(
                        batcher(
                            model,
                            tokenizer,
                            sentences[i : i + batch_size],
                            args,
                            device="cuda",
                        )
                    )
                sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

            # sentence_embeddings = batcher(model, tokenizer, sentences, args, device="cuda")
            if args.verbose:
                print("clutering...")
            kmeans_model = KMeans(
                n_clusters=len(set(labels)), random_state=args.seed
            ).fit(sentence_embeddings)
            pred_labels = kmeans_model.labels_

            if dataset_key == "WN-unified":
                datasets = []
                mapping_dict = get_label_mapping_dict(pred_labels, labels)
                pred_labels = [mapping_dict[pred_label] for pred_label in pred_labels]

                for lang, start_idx, end_idx in lang_pos:
                    acc = (
                        np.sum(
                            (
                                np.array(labels[start_idx:end_idx])
                                == pred_labels[start_idx:end_idx]
                            )
                        )
                        / len(pred_labels[start_idx:end_idx])
                        * 100
                    )
                    # acc = accuracy(labels[start_idx:end_idx], pred_labels[start_idx:end_idx]) * 100
                    scores.append("%.2f" % (acc))
                    mlflow_writer.log_metric(f"{dataset_key}-{lang}", acc)
                    datasets.append(f"{dataset_key}-{lang}")
                break

            acc = accuracy(labels, pred_labels) * 100
            if args.verbose:
                print("%.2f" % (acc))
            scores.append("%.2f" % (acc))
            mlflow_writer.log_metric(dataset_key, acc)

        datasets.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        results.append((datasets, scores))
        mlflow_writer.log_metric(
            f"{datasets[0]}-Avg.", sum([float(score) for score in scores]) / len(scores)
        )

    for result in results:
        datasets, scores = result
        print("------ %s ------" % (datasets[0]))
        print_table(datasets, scores)


if __name__ == "__main__":
    main()
