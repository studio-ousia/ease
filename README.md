## EASE: Entity-Aware Contrastive Learning of Sentence Embedding

<!-- TODO add license -->
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-pink?color=FF33CC)](https://github.com/huggingface/transformers)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/sosuke)
[![Arxiv](https://img.shields.io/badge/arXiv-2205.04260-B21A1B)](https://arxiv.org/abs/2205.04260)

EASE is a novel method for learning sentence embeddings via contrastive learning between sentences and their related entities proposed in our paper [EASE: Entity-Aware Contrastive Learning of Sentence Embedding](https://arxiv.org/abs/2205.04260).
This repository contains the source code to train the model and evaluate it with downstream tasks.
Our code is mainly based on that of [SimCSE](https://github.com/princeton-nlp/SimCSE).

<p align="center">
<img src="figure/ease.png" width="70%">
</p>

## Released Models
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/sosuke)

Our published models are listed as follows.
You can use these models by using [HuggingFace's Transformers](https://github.com/huggingface/transformers).



|              **Monolingual Models**             | **Avg. STS** | **Avg. STC** |
|:-------------------------------|:--------:|:--------:|
|  [sosuke/ease-bert-base-uncased](https://huggingface.co/sosuke/ease-bert-base-uncased) |   77.0 |  63.1    |
| [sosuke/ease-roberta-base](https://huggingface.co/sosuke/ease-roberta-base) |  76.8 |  58.6   |
|              **Multilingual Models**              | **Avg. mSTS** | **Avg. mSTC** |
|  [sosuke/ease-bert-base-multilingual-cased](https://huggingface.co/sosuke/ease-bert-base-multilingual-cased)  |   57.2  | 36.1 |
|     [sosuke/ease-xlm-roberta-base](https://huggingface.co/sosuke/ease-xlm-roberta-base)     |   57.1 | 36.3 |


## Use EASE with Huggingface

```python

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Import our pretrained model. 
tokenizer = AutoTokenizer.from_pretrained("sosuke/ease-bert-base-multilingual-cased")
model = AutoModel.from_pretrained("sosuke/ease-bert-base-multilingual-cased")

# Set pooler.
pooler = lambda last_hidden, att_mask: (last_hidden * att_mask.unsqueeze(-1)).sum(1) / att_mask.sum(-1).unsqueeze(-1)

# Tokenize input texts.
texts = [
    "Ils se préparent pour un spectacle à l'école.",
    "They are preparing for a show at school.",
    "Two medical professionals in green look on at something."
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    last_hidden = model(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state
embeddings = pooler(last_hidden, inputs["attention_mask"])

# Calculate cosine similarities
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

print(f"Cosine similarity between {texts[0]} and {texts[1]} is {cosine_sim_0_1}")
print(f"Cosine similarity between {texts[0]} and {texts[2]} is {cosine_sim_0_2}")
```

Please see [here](https://github.com/studio-ousia/ease/blob/main/ease/ease_models.py#L109) for other pooling methods.


## Setups

[![Python](https://img.shields.io/badge/python-3.7.6-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-376/)
<!-- Python 3.7.6 -->

Run the following script to install the dependent libraries.
```bash
pip install -r requirements.txt
```

Before training, please download the datasets for training and evaluation.
```bash
bash download_all.sh
```


## Evaluation

<!-- TODO prepare for colab codes -->

We provide evaluation code for sentence embeddings including Semantic Textual Similarity ([STS 2012-2016](https://aclanthology.org/S16-1081/), [STS Benchmark](https://aclanthology.org/S17-2001/), [SICK-elatedness](http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf), and [the extended version of STS 2017 dataset](https://aclanthology.org/2020.emnlp-main.365/)), Short Text Clustering ([Eight STC benchmarks](https://aclanthology.org/2021.emnlp-main.467/) and [MewsC-16](#mewsc-16)), Cross-lingual Parallel Matching ([Tatoeba](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00288/43523/Massively-Multilingual-Sentence-Embeddings-for)) and Cross-lingual Text Classification ([MLDoc](https://aclanthology.org/L18-1560/)).

Set your model or path of tranformers-based checkpoint (`--model_name_or_path`),
pooling method type (`--pooler`), and what set of tasks (`--task_set`).
See the example code below.

##### Semantic Textual Similarity
```bash
python evaluation.py \
    --model_name_or_path sosuke/ease-bert-base-multilingual-cased \ 
    --pooler avg \ 
    --task_set cl-sts 
```

##### Short Text Clustering
```bash
python downstreams/text-clustering/evaluation.py \
    --model_name_or_path sosuke/ease-bert-base-multilingual-cased \
    --pooler avg \ 
    --task_set cl
```

##### Cross-lingual Parallel Matching
```bash
python downstreams/parallel-matching/evaluation.py \
    --model_name_or_path sosuke/ease-bert-base-multilingual-cased \ 
    --pooler avg 
```

##### Cross-lingual Text Classification
```bash
python downstreams/cross-lingual-transfer/evaluation.py \
    --model_name_or_path sosuke/ease-bert-base-multilingual-cased \ 
    --pooler avg
```

Please refer to each evaluation code for detailed descriptions of arguments.


## Training


You can train an EASE model in a monolingual setting using English Wikipedia sentences or in a multilingual setting using Wikipedia sentences in 18 languages.

We provide example trainig scripts for both monolingual ([train_monolingual_ease.sh](https://github.com/studio-ousia/ease/blob/main/train_monolingual_ease.sh)) and multilingual ([train_multilingual_ease.sh](https://github.com/studio-ousia/ease/blob/main/train_multilingual_ease.sh)) settings.


## MewsC-16

We construct [MewsC-16](https://github.com/studio-ousia/ease/tree/main/downstreams/text-clustering/data/mewsc16) (**M**ultilingual Short Text **C**lustering Dataset for N**ews** in **16** languages) from Wikinews.
This dataset contains topic sentences from Wikinews articles in 13 categories and 16 languages. More detailed information is available in our paper, Appendix E.

<!-- TODO reproduction code -->
##### Statistics and Scores

|              **Language**             | **Sentences** | **Label types** |**XLM-R<sub>base</sub>** |**EASE-XLM-R<sub>base</sub>** |
|:--------:|--------:|--------:|--------:|--------:|
| ar | 2,224 | 11 | 27.9 | 27.4
| ca | 3,310 | 11 | 27.1 | 27.9
| cs | 1,534 | 9 | 25.2 | 41.2
| de | 6,398 | 8 | 30.5 | 39.5
| en | 12,892 | 13 | 25.8 | 39.6
| eo | 227 | 8 | 24.7 | 37.0
| es | 6,415 | 11 | 20.8 | 38.2
| fa | 773 | 9 | 37.2 | 41.5
| fr | 10,697 | 13 | 25.3 | 33.3
| ja | 1,984 | 12 | 44.0 | 47.6
| ko | 344 | 10 | 24.1 | 33.7
| pl | 7,247 | 11 | 28.8 | 39.9
| pt | 8,921 | 11 | 27.4 | 32.9
| ru | 1,406 | 12 | 20.1 | 27.2
| sv | 584 | 7 | 30.1 | 29.8
| tr | 459 | 7 | 30.7 | 44.9
| Avg. |  |  | 28.1 | 36.3

Note that the results are slightly different from those reported in the original paper since we further cleaned the data after the publication.

## Citation
[![Arxiv](https://img.shields.io/badge/arXiv-2205.04260-B21A1B)](https://arxiv.org/abs/2205.04260)

```bibtex
@inproceedings{nishikawa-etal-2022-ease,
    title = "{EASE}: Entity-Aware Contrastive Learning of Sentence Embedding",
    author = "Nishikawa, Sosuke  and
      Ri, Ryokan  and
      Yamada, Ikuya  and
      Tsuruoka, Yoshimasa  and
      Echizen, Isao",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.284",
    pages = "3870--3885",
    abstract = "We present EASE, a novel method for learning sentence embeddings via contrastive learning between sentences and their related entities.The advantage of using entity supervision is twofold: (1) entities have been shown to be a strong indicator of text semantics and thus should provide rich training signals for sentence embeddings; (2) entities are defined independently of languages and thus offer useful cross-lingual alignment supervision.We evaluate EASE against other unsupervised models both in monolingual and multilingual settings.We show that EASE exhibits competitive or better performance in English semantic textual similarity (STS) and short text clustering (STC) tasks and it significantly outperforms baseline methods in multilingual settings on a variety of tasks.Our source code, pre-trained models, and newly constructed multi-lingual STC dataset are available at https://github.com/studio-ousia/ease.",
}
```
