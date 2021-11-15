def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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
from omegaconf import OmegaConf
from utils.mlflow_writer import MlflowWriter


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

    parser.add_argument('--verbose', action='store_true')

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # datasets = ["tweet", "biomedical", "stackoverflow", "searchsnippets", "agnews", "googlenews_ts", "googlenews_t", "googlenews_s"] 
    # datasets = []
    dataset_groups = []
    if args.task_set in ["full", "mono"]:
        dataset_groups.append(["R8", "R52", "OH", '20N'])
        dataset_groups.append(['AG', 'SS', 'SO', 'Bio', 'Tweet', 'G-TS', 'G-S', 'G-T'])
    
    if args.task_set in ["full", "cl"]:

        # 15 langs label-unified 
        # dataset_groups.append(['WN-TFS-ar', 'WN-TFS-ca', 'WN-TFS-cs', 'WN-TFS-de', 'WN-TFS-es', 'WN-TFS-eo', 'WN-TFS-fa', 'WN-TFS-fr', 'WN-TFS-ko', 'WN-TFS-ja', 'WN-TFS-pl', 'WN-TFS-pt', 'WN-TFS-ru', 'WN-TFS-sv', 'WN-TFS-tr'])
        # dataset_groups.append(['WN-FS-ar', 'WN-FS-ca', 'WN-FS-cs', 'WN-FS-de', 'WN-FS-es', 'WN-FS-eo', 'WN-FS-fa', 'WN-FS-fr', 'WN-FS-ko', 'WN-FS-ja', 'WN-FS-pl', 'WN-FS-pt', 'WN-FS-ru', 'WN-FS-sv', 'WN-FS-tr'])
        dataset_groups.append(['WN-S-ar', 'WN-S-ca', 'WN-S-cs', 'WN-S-de', 'WN-S-es', 'WN-S-eo', 'WN-S-fa', 'WN-S-fr', 'WN-S-ko', 'WN-S-ja', 'WN-S-pl', 'WN-S-pt', 'WN-S-ru', 'WN-S-sv', 'WN-S-tr'])
        dataset_groups.append(['WN-T-ar', 'WN-T-ca', 'WN-T-cs', 'WN-T-de', 'WN-T-es', 'WN-T-eo', 'WN-T-fa', 'WN-T-fr', 'WN-T-ko', 'WN-T-ja', 'WN-T-pl', 'WN-T-pt', 'WN-T-ru', 'WN-T-sv', 'WN-T-tr'])
        dataset_groups.append(['WN-TS-ar', 'WN-TS-ca', 'WN-TS-cs', 'WN-TS-de', 'WN-TS-es', 'WN-TS-eo', 'WN-TS-fa', 'WN-TS-fr', 'WN-TS-ko', 'WN-TS-ja', 'WN-TS-pl', 'WN-TS-pt', 'WN-TS-ru', 'WN-TS-sv', 'WN-TS-tr'])
        # dataset_groups.append(['WN-S-en', 'WN-S-ja', 'WN-S-es', 'WN-S-tr', 'WN-S-ar'])
        # dataset_groups.append(['WN-T-en', 'WN-T-de', 'WN-T-fr'])

#         dataset_groups.append(['WN-S-ar'])
#         dataset_groups.append(['WN-TS-ar'])
        # 13 langs
        # dataset_groups.append(['WN-T-en', 'WN-T-ar', 'WN-T-ja', 'WN-T-es', 'WN-T-tr', 'WN-T-it', 'WN-T-ko', 'WN-T-pl', 'WN-T-fa', 'WN-T-nl', 'WN-T-ru'])
        # dataset_groups.append(['WN-S-en', 'WN-S-ar', 'WN-S-ja', 'WN-S-es', 'WN-S-tr', 'WN-S-it', 'WN-S-ko', 'WN-S-pl', 'WN-S-fa', 'WN-S-nl', 'WN-S-ru'])
        # dataset_groups.append(['WN-TS-en', 'WN-TS-ar', 'WN-TS-ja', 'WN-TS-es', 'WN-TS-tr', 'WN-TS-it', 'WN-TS-ko', 'WN-TS-pl', 'WN-TS-fa', 'WN-TS-nl', 'WN-TS-ru'])
        
        # NC
        # dataset_groups.append(["NC-en", "NC-de", "NC-es"])

        # dataset_groups.append(['WN-T-en', 'WN-T-ar', 'WN-T-ja', 'WN-T-es', 'WN-T-tr', 'WN-T-it', 'WN-T-ko', 'WN-T-pt', 'WN-T-uk', 'WN-T-cs', 'WN-T-pl', 'WN-T-ca', 'WN-T-fi', 'WN-T-fa', 'WN-T-nl', 'WN-T-hu', 'WN-T-eo', 'WN-T-ru'])
        # dataset_groups.append(['WN-S-en', 'WN-S-ar', 'WN-S-ja', 'WN-S-es', 'WN-S-tr', 'WN-S-it', 'WN-S-ko', 'WN-S-pt', 'WN-S-uk', 'WN-S-cs', 'WN-S-pl', 'WN-S-ca', 'WN-S-fi', 'WN-S-fa', 'WN-S-nl', 'WN-S-hu', 'WN-S-eo', 'WN-S-ru'])
        # dataset_groups.append(['WN-TS-en', 'WN-TS-ar', 'WN-TS-ja', 'WN-TS-es', 'WN-TS-tr', 'WN-TS-it', 'WN-TS-ko', 'WN-TS-pt', 'WN-TS-uk', 'WN-TS-cs', 'WN-TS-pl', 'WN-TS-ca', 'WN-TS-fi', 'WN-TS-fa', 'WN-TS-nl', 'WN-TS-hu', 'WN-TS-eo', 'WN-TS-ru'])
        # datasets += ['WN-S-en']
        # datasets += ['WN-en', 'WN-ar', 'WN-ja', 'WN-es', 'WN-tr', 'WN-it', 'WN-ko', 'WN-pt', 'WN-uk', 'WN-cs', 'WN-pl', 'WN-ca', 'WN-fi', 'WN-fa', 'WN-nl', 'WN-hu', 'WN-eo']
        # datasets += ['WN-en', 'WN-ar', 'WN-ja', 'WN-es', 'WN-tr', 'WN-it', 'WN-ko', 'WN-pt', 'WN-uk', 'WN-cs', 'WN-pl', 'WN-ca', 'WN-fi', 'WN-fa', 'WN-nl', 'WN-hu', 'WN-eo', 'WN-ru']
        # datasets += ['MD-en', 'MD-fr', 'MD-de', 'MD-ja', 'MD-zh', 'MD-it', 'MD-ru', 'MD-es']
        # datasets += ['MD-FS-en', 'MD-FS-fr', 'MD-FS-de', 'MD-FS-ja', 'MD-FS-zh', 'MD-FS-it', 'MD-FS-ru', 'MD-FS-es']
    batch_size = 256

    results = []

    for datasets in dataset_groups:
        scores = []
        for dataset_key in tqdm(datasets):
            if args.verbose:
                print(f"evaluate {dataset_key}...")
            sentences, labels = dataset_load(dataset_key)

            if args.verbose:
                print("encode sentence embeddings...")
            sentence_embeddings = []
            for i in tqdm(range(0, len(sentences), batch_size)):
                sentence_embeddings.append(batcher(model, tokenizer, sentences[i: i + batch_size], args, device="cuda"))
            sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

            # sentence_embeddings = batcher(model, tokenizer, sentences, args, device="cuda")
            if args.verbose:
                print("clutering...")
            kmeans_model = KMeans(n_clusters=len(set(labels)), random_state=args.seed).fit(sentence_embeddings)
            pred_labels = kmeans_model.labels_
            acc = accuracy(labels, pred_labels) * 100
            if args.verbose:
                print("%.1f" % (acc))
            scores.append("%.1f" % (acc))
            mlflow_writer.log_metric(dataset_key, acc)

        datasets.append("Avg.")
        scores.append("%.1f" % (sum([float(score) for score in scores]) / len(scores)))
        results.append((datasets, scores))
        mlflow_writer.log_metric(f"{datasets[0]}-Avg.", sum([float(score) for score in scores]) / len(scores))
        
    for result in results:
        datasets, scores = result
        print("------ %s ------" % (datasets[0]))
        print_table(datasets, scores)


if __name__ == "__main__":
    main()
