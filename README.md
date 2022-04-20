## EASE: Entity-Aware Contrastive Learning of Sentence Embedding

EASE is a novel method for learning sentence embeddings via contrastive learning between sentences and their related entities proposed in out paper EASE: Entity-Aware Contrastive Learning of Sentence Embedding.
This repository contains the source code to train the model and evaluate it with downstream tasks.

<img src="figure/ease.png" width="70%">


## Quick Links


  - [Model List](#model-list)
  - [Training](#training)
    - [Preparation](#preparation)
  - [Evaluation](#evaluation)
    - [Semantic Textual Similarity](#sts)
    - [Short Text Clustering](#stc)
    - [Cross-lingual Parallel Matching](#clpm)
    - [Cross-lingual Text Classification](#cltc)
  - [MewsC-16](#mewsc-16)
  - [Citation](#citation)

## Model List

Our published models are listed as follows.



|              **Monolingual Models**             | **Avg. STS** | **Avg. STC** |
|:-------------------------------|:--------:|:--------:|
|  [sosuke/ease-bert-base-uncased](https://huggingface.co/sosuke/ease-bert-base-uncased) |   77.0 |  63.1    |
| [sosuke/ease-roberta-base](https://huggingface.co/sosuke/ease-roberta-base) |  76.8 |  58.6   |
|              **Multilingual Models**              | **Avg. mSTS** | **Avg. mSTC** |
|  [sosuke/ease-bert-base-multilingual-cased](https://huggingface.co/princeton-nlp/sup-simcse-bert-large-uncased)  |   57.2  | 36.0 |
|     [sosuke/ease-xlm-roberta-base](https://huggingface.co/sosuke/ease-xlm-roberta-base)     |   57.1 | 35.4 |


## Training

##### Preparation

Run the following script to install the dependent libraries.
```bash
pip install -r requirements.txt
```

Before training, please download the training and evaluation datasets.
```bash
bash download_all.sh
```
<!-- TODO check download_all.sh -->

Please see the example scripts to train an EASE model.

Hydra利用しているよ
config見てね

## Evaluation



##### Semantic Textual Similarity

```bash
python evaluation.py \
    --model_name_or_path bert-base-uncased \ # set your model path
    --pooler avg \ # set pooling method
    --task_set sts \ # sts or cl-sts o
    --mode test
```

##### Short Text Clustering

```bash
python text-clustering/evaluate.py \
    --model_name_or_path bert-base-uncased \ # set your model path
    --pooler avg \ # set pooling method
    --task_set mono # mono or cl or full
```

##### Cross-lingual Parallel Matching

```bash
python parallel-matching/similar_sentence_search.py \
    --model_name_or_path bert-base-multilingual-cased \ # set your model path
    --pooler avg # set pooling method
```

##### Cross-lingual Text Classification

```bash
cd cross-lingual-transfer/data
download_dataset.sh
```

```bash
python cross-lingual-transfer/main.py
    --model_name_or_path bert-base-multilingual-cased \ # set your model path
    --pooler avg \ # set pooling method
```

## MewsC-16

We constructed [MewsC-16](https://github.com/Sosuke115/EASE/tree/main/text-clustering/data/mewsc16) (**M**ultilingual Short Text **C**lustering Dataset for N**ews** in **16** languages) from Wikinews.
The dataset contains topic sentences from Wikinews articles in 13 categories and 16 languages. More detailed information is available in our paper, Appendix E.

<!-- TODO reproduction code -->

##### Statistics

<table align="left">
  <tr>
    <td>1番目</td>
  </tr>
</table>
<table>
  <tr>
    <td>2番目</td>
  </tr>
</table>

| Language | Code | Docs  | Mentions | Unique Entities | Entities outside En-Wiki |
|----------|------|-------|----------|-----------------|--------------------------|
| Japanese | ja   | 3410  | 34463    | 13663           | 3384                     |
| German   | de   | 13703 | 65592    | 23086           | 3054                     |
| Spanish  | es   | 10284 | 56716    | 22077           | 1805                     |
| Arabic   | ar   | 1468  | 7367     | 2232            | 141                      |
| Serbian  | sr   | 15011 | 35669    | 4332            | 269                      |
| Turkish  | tr   | 997   | 5811     | 2630            | 157                      |
| Persian  | fa   | 165   | 535      | 385             | 12                       |
| Tamil    | ta   | 1000  | 2692     | 1041            | 20                       |
| English  | en   | 12679 | 80242    | 38697           | 14                       |
| Total    |      | 58717 | 289087   | 82162           | 8807                     |


## Citation
[TBA]

<!-- TODO -->