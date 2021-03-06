import argparse
import io
import logging
import os
import re
import sys

import numpy as np
import torch
import transformers
from omegaconf import OmegaConf
from prettytable import PrettyTable
from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer

from utils.mlflow_writer import MlflowWriter
from utils.utils import get_mlflow_writer, print_table

# Set up logger
logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = "./SentEval"
PATH_TO_DATA = "./SentEval/data"

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Transformers' model name or path",
    )
    parser.add_argument(
        "--pooler",
        type=str,
        choices=["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        default="cls",
        help="Which pooler to use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "test", "fasttest", "align_uniform"],
        default="test",
        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results",
    )
    parser.add_argument(
        "--task_set",
        type=str,
        choices=["sts", "transfer", "full", "na", "cl-sts"],
        default="sts",
        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "MR",
            "CR",
            "MPQA",
            "SUBJ",
            "SST2",
            "TREC",
            "MRPC",
            "SICKRelatedness",
            "STSBenchmark",
        ],
        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="evals",
        help="mlflow experiment name",
    )

    args = parser.parse_args()

    # mlflow
    mlflow_writer = get_mlflow_writer(args.experiment_name, "mlruns", OmegaConf.create({"eval_args": vars(args)}))

    # Load transformers' model checkpoint
    print("model_path", args.model_name_or_path)
    # return
    model = AutoModel.from_pretrained(args.model_name_or_path)
    if "xlm" in args.model_name_or_path:
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set up the tasks
    if args.task_set == "sts":
        args.tasks = ["STSBenchmark", "SICKRelatedness"]
        if args.mode == "test" or args.mode == "fasttest":
            args.tasks += ["STS12", "STS13", "STS14", "STS15", "STS16"]
    elif args.task_set == "transfer":
        args.tasks = ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
    elif args.task_set == "cl-sts":
        # args.tasks = ['STS16CL', 'STS17']
        args.tasks = ["STS17"]
    elif args.task_set == "full":
        args.tasks = [
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "STSBenchmark",
            "SICKRelatedness",
        ]
        args.tasks += ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]

    # Set params for SentEval
    if args.mode == "dev" or args.mode == "fasttest":
        # Fast mode
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 5}
        params["classifier"] = {
            "nhid": 0,
            "optim": "rmsprop",
            "batch_size": 32,
            "tenacity": 3,
            "epoch_size": 2,
        }
    elif args.mode == "test":
        # Full mode
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 10}
        params["classifier"] = {
            "nhid": 0,
            "optim": "adam",
            "batch_size": 32,
            "tenacity": 5,
            "epoch_size": 4,
        }
    elif args.mode == "align_uniform":
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 10}
    else:
        raise NotImplementedError

    print(args.tasks)

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode("utf-8") for word in s] for s in batch]

        sentences = [" ".join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors="pt",
                padding=True,
                max_length=max_length,
                truncation=True,
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
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
                (first_hidden + last_hidden)
                / 2.0
                * batch["attention_mask"].unsqueeze(-1)
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

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == "dev":
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ["STSBenchmark", "SICKRelatedness"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["dev"]["spearman"][0] * 100))
                mlflow_writer.log_metric(
                    f"{task}-alignment", results[task]["dev"]["align_loss"]
                )
                mlflow_writer.log_metric(
                    f"{task}-uniformity", results[task]["dev"]["uniform_loss"]
                )
            else:
                scores.append("0.00")

        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["devacc"]))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == "align_uniform":
        task_names = []
        scores = []
        datasets = [
            "STS.input.track5.en-en.txt",
            "STS.input.track7.en-de.txt",
            "STS.input.track8.fr-en.txt",
            "STS.input.track9.it-en.txt",
            "STS.input.track10.nl-en.txt",
        ]

        print(results[task].keys())

        for dataset in datasets:
            lang_name = re.findall("STS.input.track\d+.?\.(.+).txt", dataset)[0]
            if task in results:
                scores.append(
                    "%.2f" % (results[task][dataset]["spearman"].correlation * 100)
                )
                mlflow_writer.log_metric(
                    lang_name, results[task][dataset]["spearman"].correlation * 100
                )
                mlflow_writer.log_metric(
                    f"{lang_name}-align", results[task][dataset]["align_loss"]
                )
                mlflow_writer.log_metric(
                    f"{lang_name}-uniform", results[task][dataset]["uniform_loss"]
                )
            else:
                scores.append("0.00")
            task_names.append(lang_name)

        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        mlflow_writer.log_metric(
            "CL-STS Avg.", sum([float(score) for score in scores]) / len(scores)
        )
        print_table(task_names, scores)

    elif args.mode == "test" or args.mode == "fasttest":
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in [
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "STSBenchmark",
            "SICKRelatedness",
        ]:
            task_names.append(task)
            if task in results:
                if task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                    scores.append(
                        "%.2f" % (results[task]["all"]["spearman"]["all"] * 100)
                    )
                    mlflow_writer.log_metric(
                        task, results[task]["all"]["spearman"]["all"] * 100
                    )
                else:
                    scores.append(
                        "%.2f" % (results[task]["test"]["spearman"].correlation * 100)
                    )
                    mlflow_writer.log_metric(
                        task, results[task]["test"]["spearman"].correlation * 100
                    )
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        mlflow_writer.log_metric(
            "STS-Avg.", sum([float(score) for score in scores]) / len(scores)
        )
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["acc"]))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        ## cross-lingual
        task_names = []
        scores = []

        # for task in ['STS16CL', 'STS17']:
        for task in ["STS17"]:
            if task == "STS17":

                alignments = []
                uniformities = []

                datasets = [
                    "STS.input.track5.en-en.txt",
                    "STS.input.track1.ar-ar.txt",
                    "STS.input.track3.es-es.txt",
                    "STS.input.track2.ar-en.txt",
                    "STS.input.track7.en-de.txt",
                    "STS.input.track6.tr-en.txt",
                    "STS.input.track4a.es-en.txt",
                    "STS.input.track8.fr-en.txt",
                    "STS.input.track9.it-en.txt",
                    "STS.input.track10.nl-en.txt",
                ]

                for dataset in datasets:
                    lang_name = re.findall("STS.input.track\d+.?\.(.+).txt", dataset)[0]
                    if task in results:
                        mlflow_writer.log_metric(
                            lang_name,
                            results[task][dataset]["spearman"].correlation * 100,
                        )
                        scores.append(
                            "%.2f"
                            % (results[task][dataset]["spearman"].correlation * 100)
                        )

                        alignments.append(results[task][dataset]["align_loss"])
                        uniformities.append(results[task][dataset]["uniform_loss"])
                    else:
                        scores.append("0.00")
                    task_names.append(lang_name)

                mlflow_writer.log_metric("align", np.mean(alignments))
                mlflow_writer.log_metric("uniform", np.mean(uniformities))

            else:
                task_names.append(task)
                if task in results:
                    scores.append(
                        "%.2f" % (results[task]["all"]["spearman"]["all"] * 100)
                    )
                    mlflow_writer.log_metric(
                        task, results[task]["all"]["spearman"]["all"] * 100
                    )
                else:
                    scores.append("0.00")

        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        mlflow_writer.log_metric(
            "CL-STS Avg.", sum([float(score) for score in scores]) / len(scores)
        )
        print_table(task_names, scores)


if __name__ == "__main__":
    main()
