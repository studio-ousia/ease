from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    BertForPreTraining,
)
from torch.utils.data import TensorDataset, random_split

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import os
from tqdm import tqdm
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AdamW,
    XLMRobertaTokenizer,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from wikipedia2vec import Wikipedia2Vec

from ease.ease_models import BertForEACL, RobertaForEACL
from ease.trainers import CLTrainer

from utils.utils import pickle_dump, pickle_load, save_model
from utils.mlflow_writer import MlflowWriter

import pprint
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import mlflow

from dataset import MyDataset, get_dataset

from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

from transformers import (
    default_data_collator,
    TrainingArguments,
)

import gc


from transformers.file_utils import (
    cached_property,
    torch_required,
    is_torch_available,
    is_torch_tpu_available,
)
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    PreTrainedTokenizerBase,
)


ENTITY_PAD_MARK = "[PAD]"


@dataclass
class OurTrainingArguments(TrainingArguments):
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."},
    )

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
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


def update_args(base_args, input_args):
    for key, value in dict(input_args).items():
        base_args.__dict__[key] = value
    return base_args


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    model_args, train_args = cfg.model_args, cfg.train_args

    # TODO refactoring
    parser = HfArgumentParser(OurTrainingArguments)
    # base_train_args = parser.parse_args_into_dataclasses()
    # # parser = HfArgumentParser(TrainingArguments)
    base_train_args = parser.parse_args_into_dataclasses(
        ["--output_dir", "saved_models", "--evaluation_strategy", "steps"]
    )[0]
    train_args = update_args(base_train_args, train_args)
    train_args.output_dir = os.path.join(cwd, train_args.output_dir)

    EXPERIMENT_NAME = train_args.experiment_name
    tracking_uri = f"file:{cwd}/mlruns"
    mlflow_writer = MlflowWriter(EXPERIMENT_NAME, tracking_uri=tracking_uri)
    mlflow_writer.log_params_from_omegaconf_dict(cfg)

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.cuda.manual_seed_all(train_args.seed)

    # トークナイザ
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": True,
        "revision": "main",
        "use_auth_token": True if False else None,
    }

    if "xlm" in model_args.model_name_or_path:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )

    print("loading data...")
    wikipedia_data = []

    # wiki_en or wiki_18 or your dataset path
    if train_args.dataset_name_or_path == "wiki_en":
        wikipedia_data = load_dataset(
            "json", data_files=os.path.join(cwd, "data/ease-dataset-en.json")
        )["train"]
    elif train_args.dataset_name_or_path == "wiki_18":
        wikipedia_data = load_dataset(
            "json", data_files=os.path.join(cwd, "data/ease-dataset-18-langs.json")
        )["train"]
    elif train_args.dataset_name_or_path == "test":
        wikipedia_data = load_dataset(
            "json", data_files=os.path.join(cwd, "data/ease-dataset-test.json")
        )["train"]

    else:
        # TODO load from dataset path
        raise NotImplementedError()

    # build entity vocab
    print("build entity vocab...")
    if train_args.train_saved_model:
        entity_vocab = pickle_load(
            os.path.join(model_args.model_name_or_path, "entity_vocab.pkl")
        )
    else:
        entity_vocab = build_entity_vocab(wikipedia_data)

    print(f"entities: {len(entity_vocab)}")
    print("get_dataset...")
    input_ids, attention_masks, token_type_ids, title_id, hn_title_ids = get_dataset(
        wikipedia_data,
        model_args.max_seq_length,
        tokenizer,
        entity_vocab,
        model_args.masked_sentence_ratio,
    )
    train_dataset = MyDataset(
        input_ids,
        attention_masks,
        token_type_ids,
        title_id,
        hn_title_ids,
        model_args.model_name_or_path,
    )

    del wikipedia_data
    gc.collect()

    # load pretrained entity embeddings
    embedding = Wikipedia2Vec.load(os.path.join(cwd, train_args.wikipedia2vec_path))
    dim_size = embedding.syn0.shape[1]

    # set struct omegaconf
    OmegaConf.set_struct(model_args, True)
    with open_dict(model_args):
        model_args.entity_emb_shape = (len(entity_vocab), dim_size)

    entity_embeddings = np.random.uniform(
        low=-0.05, high=0.05, size=model_args.entity_emb_shape
    )
    print(model_args.entity_emb_shape)

    # init entity embeddings
    entity_embeddings[0] = np.zeros(dim_size)
    cnt = 0

    if model_args.init_wiki2emb:
        print("init wiki2vec")
        for entity, index in tqdm(entity_vocab.items()):
            try:
                entity_embeddings[index] = embedding.get_entity_vector(entity)
                cnt += 1
            except KeyError:
                pass
        print(cnt)

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

        if model_args.do_mlm:
            if "multilingual" in model_args.model_name_or_path:
                pretrained_model = BertForPreTraining.from_pretrained(
                    "bert-base-multilingual-cased"
                )
            else:
                pretrained_model = BertForPreTraining.from_pretrained(
                    "bert-base-uncased"
                )

            model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())

    model.resize_token_embeddings(len(tokenizer))
    model.entity_embedding.weight = nn.Parameter(torch.FloatTensor(entity_embeddings))
    model.entity_embedding.weight.requires_grad = True

    del entity_embeddings
    gc.collect()

    model_path = (
        model_args.model_name_or_path
        if (
            model_args.model_name_or_path is not None
            and os.path.isdir(model_args.model_name_or_path)
        )
        else None
    )

    # Data collator
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
            if model_args.do_mlm:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(
                    batch["input_ids"]
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

        def mask_tokens(
            self,
            inputs: torch.Tensor,
            special_tokens_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(
                        val, already_has_special_tokens=True
                    )
                    for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(
                    special_tokens_mask, dtype=torch.bool
                )
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            )
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.mask_token
            )

            # 10% of the time, we replace masked input tokens with random word
            indices_random = (
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                & masked_indices
                & ~indices_replaced
            )
            random_words = torch.randint(
                len(self.tokenizer), labels.shape, dtype=torch.long
            )
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

    trainer = CLTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset if train_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=OurDataCollatorWithPadding(tokenizer),
    )

    trainer.model_args = model_args

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

    del train_dataset
    gc.collect()

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

    del trainer, model, tokenizer
    gc.collect()


if __name__ == "__main__":
    main()
