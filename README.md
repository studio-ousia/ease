## EASE: Entity-Aware Contrastive Learning of Sentence Embedding

## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
  - [Model List](#model-list)
  - [Use SimCSE with Huggingface](#use-simcse-with-huggingface)
  - [Train SimCSE](#train-simcse)
    - [Requirements](#requirements)
    - [Evaluation](#evaluation)
    - [Training](#training)
  - [Citation](#citation)

## Overview 

![](figure/ease.png)

## Preparation

<!-- TODO wiki2vecの追加 -->

```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

## Model List

Our published models are listed as follows.



|              **Monolingual Models**             | **Avg. STS** | **Avg. STC** |
|:-------------------------------|:--------:|:--------:|
|  [sosuke/ease-bert-base-uncased](https://huggingface.co/sosuke/ease-bert-base-uncased) |   77.0 |  63.1    |
| [sosuke/ease-roberta-base](https://huggingface.co/sosuke/ease-roberta-base) |  76.8 |  58.6   |
|              **Multilingual Models**              | **Avg. mSTS** | **Avg. mSTC** |
|  [sosuke/ease-bert-base-multilingual-cased](https://huggingface.co/princeton-nlp/sup-simcse-bert-large-uncased)  |   57.2  | 36.0 |
|     [sosuke/ease-xlm-roberta-base](https://huggingface.co/sosuke/ease-xlm-roberta-base)     |   57.1 | 35.4 |

### Semantic textual similarity

```bash
python evaluation.py \
    --model_name_or_path bert-base-uncased \ # set your model path
    --pooler avg \ # set pooling method
    --task_set sts \ # sts or cl-sts o
    --mode test
```

### Short text clustering

```bash
python text-clustering/evaluate.py \
    --model_name_or_path bert-base-uncased \ # set your model path
    --pooler avg \ # set pooling method
    --task_set mono # mono or cl or full
```

### Cross-lingual Parallel Matching

```bash
python parallel-matching/similar_sentence_search.py \
    --model_name_or_path bert-base-multilingual-cased \ # set your model path
    --pooler avg # set pooling method
```

### Cross-lingual Zero-shot Classification

```bash
cd cross-lingual-transfer/data
download_dataset.sh
```

```bash
python cross-lingual-transfer/main.py
    --model_name_or_path bert-base-multilingual-cased \ # set your model path
    --pooler avg \ # set pooling method
```
