from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from transformers import MODEL_FOR_MASKED_LM_MAPPING, BertForPreTraining

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import logging
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, XLMRobertaTokenizer
from wikipedia2vec import Wikipedia2Vec

from dataset import get_dataset
from ease.ease_models import BertForEACL, RobertaForEACL
from ease.trainers import CLTrainer
from utils.utils import get_mlflow_writer, pickle_dump, pickle_load, set_seeds

logger = logging.getLogger(__name__)

from transformers import TrainingArguments
from transformers.file_utils import cached_property, torch_required
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)

ENTITY_PAD_MARK = "[PAD]"


@dataclass
class OurTrainingArguments(TrainingArguments):
    resume_from_checkpoint: bool = field(
        default=False,
    )
    group_by_length: bool = field(
        default=False,
    )
    eval_transfer: bool = field(
        default=False,
    )

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif self.local_rank == -1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._n_gpu = torch.cuda.device_count()
        else:
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError(
                        "--deepspeed requires deepspeed: `pip install deepspeed`."
                    )
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


# data collator
@dataclass
class OurDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        special_keys = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "mlm_input_ids",
            "mlm_labels",
        ]
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]["input_ids"])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append(
                    {
                        k: feature[k][i] if k in special_keys else feature[k]
                        for k in feature
                    }
                )

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {
            k: batch[k].view(bs, num_sent, -1)
            if k in special_keys
            else batch[k].view(bs, num_sent, -1)[:, 0]
            for k in batch
        }

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


def build_entity_vocab(data):
    entities = []
    for d in tqdm(data):
        title, hn_titles = d["positive_entity"], d["negative_entity"]
        entities.append(title)
        entities.extend(hn_titles)
    entities = set(entities) - set([ENTITY_PAD_MARK])
    entity_vocab = {ENTITY_PAD_MARK: 0}
    entity_vocab.update({title: i + 1 for i, title in enumerate(entities)})
    return entity_vocab


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    model_args, data_args, train_args = cfg.model_args, cfg.data_args, cfg.train_args
    train_args = OurTrainingArguments(**train_args)
    train_args.output_dir = os.path.join(cwd, train_args.output_dir)
    mlflow_writer = get_mlflow_writer(
        data_args.experiment_name, f"file:{cwd}/mlruns", cfg
    )

    set_seeds(train_args.seed)

    # tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": True,
        "revision": "main",
        "use_auth_token": None,
    }

    if "xlm" in model_args.model_name_or_path:
        tokenizer = XLMRobertaTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )

    wikipedia_data = []

    # wiki_en or wiki_18
    if data_args.dataset_name_or_path == "wiki_en":
        wikipedia_data = load_dataset(
            "json", data_files=os.path.join(cwd, "data/ease-dataset-en.json")
        )["train"]
    elif data_args.dataset_name_or_path == "wiki_18":
        wikipedia_data = load_dataset(
            "json", data_files=os.path.join(cwd, "data/ease-dataset-18-langs.json")
        )["train"]
    elif data_args.dataset_name_or_path == "test":
        wikipedia_data = load_dataset("sosuke/ease-dataset-test.json")["train"]
    else:
        # TODO load from your dataset
        raise NotImplementedError()

    # build entity vocab
    if train_args.resume_from_checkpoint:
        entity_vocab = pickle_load(
            os.path.join(model_args.model_name_or_path, "entity_vocab.pkl")
        )
    else:
        entity_vocab = build_entity_vocab(wikipedia_data)

    # print(f"entities: {len(entity_vocab)}")

    # load pretrained entity embeddings
    embedding = Wikipedia2Vec.load(os.path.join(cwd, data_args.wikipedia2vec_path))
    dim_size = embedding.syn0.shape[1]
    OmegaConf.set_struct(model_args, True)
    with open_dict(model_args):
        model_args.entity_emb_shape = (len(entity_vocab), dim_size)
    entity_embeddings = np.random.uniform(
        low=-0.05, high=0.05, size=model_args.entity_emb_shape
    )
    entity_embeddings[0] = np.zeros(dim_size)
    cnt = 0
    if model_args.init_wiki2emb:
        for entity, index in tqdm(entity_vocab.items()):
            try:
                entity_embeddings[index] = embedding.get_entity_vector(entity)
                cnt += 1
            except KeyError:
                pass

    # model
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    if "roberta" in model_args.model_name_or_path:

        model = RobertaForEACL.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args,
        )

    elif any(model in model_args.model_name_or_path for model in ("bert", "LaBSE")):

        model = BertForEACL.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args,
        )

    model.resize_token_embeddings(len(tokenizer))
    model.init_entity_embedding(entity_embeddings)

    model_path = (
        model_args.model_name_or_path
        if (
            model_args.model_name_or_path is not None
            and os.path.isdir(model_args.model_name_or_path)
        )
        else None
    )

    train_dataset = get_dataset(
        wikipedia_data,
        model_args.max_seq_length,
        tokenizer,
        entity_vocab,
        model_args,
    )
    trainer = CLTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset if train_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=OurDataCollatorWithPadding(tokenizer),
    )

    trainer.model_args = model_args

    # training
    if train_args.do_train:
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        output_train_file = os.path.join(train_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    # evaluation
    if train_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=False)
        output_eval_file = os.path.join(train_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

        # save to Mlflow
        mlflow_writer.log_metric("eval_stsb_spearman", results["eval_stsb_spearman"])
        mlflow_writer.log_metric("eval_sickr_spearman", results["eval_sickr_spearman"])
        mlflow_writer.log_metric("eval_avg_sts", results["eval_avg_sts"])

    pickle_dump(entity_vocab, os.path.join(train_args.output_dir, "entity_vocab.pkl"))

    mlflow_writer.log_artifact(os.path.join(os.getcwd(), ".hydra/config.yaml"))
    mlflow_writer.log_artifact(os.path.join(os.getcwd(), ".hydra/hydra.yaml"))
    mlflow_writer.log_artifact(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
    mlflow_writer.log_artifact(os.path.join(os.getcwd(), "main.log"))


if __name__ == "__main__":
    main()
