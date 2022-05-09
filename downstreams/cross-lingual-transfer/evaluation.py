import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from transformers import (AutoConfig, AutoTokenizer, Trainer,
                          TrainingArguments, XLMRobertaTokenizer)

from bert import (BertForSequenceClassificationWithPooler,
                  RobertaForSequenceClassificationWithPooler)
from data import MLDocParser

sys.path.append(os.path.abspath(os.getcwd()))
from utils.mlflow_writer import MlflowWriter
from utils.utils import get_mlflow_writer, set_seeds


def load_mldoc_data(dataset_path, lang):
    data = {}
    lcode_to_lang = {
        "en": "english",
        "fr": "french",
        "de": "german",
        "ja": "japanese",
        "zh": "chinese",
        "it": "italian",
        "ru": "russian",
        "es": "spanish",
    }
    lang = lcode_to_lang[lang]
    modes = ["train.1000", "dev", "test"]
    categories = ["CCAT", "MCAT", "ECAT", "GCAT"]
    categories_index = {t: i for i, t in enumerate(categories)}
    parser = MLDocParser()
    for mode in modes:
        file_path = f"{dataset_path}/{lang}.{mode}"
        sentences, labels = zip(
            *[
                [sentence, categories_index[category]]
                for sentence, category in parser(file_path)
            ]
        )
        data[mode[:5]] = (list(sentences), list(labels))
    return data


class MLDocDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="bert-base-multilingual-cased"
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
        default="mldoc",
        help="mlflow experiment name",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-03,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.00,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--do_finetune", action="store_true")

    args = parser.parse_args()
    
    set_seeds(args.seed)

    mlflow_writer = get_mlflow_writer(args.experiment_name, "mlruns", OmegaConf.create({"eval_args": vars(args)}))

    dataset_path = "downstreams/cross-lingual-transfer/data"
    eval_data = dict()
    eval_langs = ["en", "fr", "de", "ja", "zh", "it", "ru", "es"]
    for lang in eval_langs:
        print(lang)
        eval_data[lang] = load_mldoc_data(dataset_path, lang)

    train_texts, train_labels = eval_data["en"]["train"]
    val_texts, val_labels = eval_data["en"]["dev"]

    if "xlm" in args.model_name_or_path:
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = MLDocDataset(train_encodings, train_labels)
    val_dataset = MLDocDataset(val_encodings, val_labels)

    config = AutoConfig.from_pretrained(args.model_name_or_path)

    config.pooler_type = args.pooler
    config.num_labels = 4
    config.do_finetune = args.do_finetune
    config.problem_type = None

    if "xlm" in args.model_name_or_path:
        model = RobertaForSequenceClassificationWithPooler.from_pretrained(
            args.model_name_or_path, config=config
        )

    elif any([name in args.model_name_or_path for name in ["bert", "LaBSE"]]):
        model = BertForSequenceClassificationWithPooler.from_pretrained(
            args.model_name_or_path, config=config
        )
    else:
        raise NotImplementedError

    if not config.do_finetune:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
            else:
                print(name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()
    print()

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=5,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=128,  # batch size for evaluation
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,
        eval_steps=10,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train()

    res = trainer.evaluate()
    print(res)
    mlflow_writer.log_metric("en-dev", res["eval_accuracy"])

    if not args.dev:
        lang_results = []

        for lang in eval_langs[1:]:
            print(lang)
            test_texts, test_labels = eval_data[lang]["test"]
            test_encodings = tokenizer(test_texts, truncation=True, padding=True)
            test_dataset = MLDocDataset(test_encodings, test_labels)
            res = trainer.evaluate(test_dataset)
            print(trainer.evaluate(test_dataset))
            lang_results.append(res["eval_accuracy"])
            mlflow_writer.log_metric(lang, res["eval_accuracy"])
        mlflow_writer.log_metric("Avg.", np.mean(lang_results))


if __name__ == "__main__":
    main()
