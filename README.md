## EASE: Entity-Aware Contrastive Learning of Sentence Embedding

## Quick Links

[WIP]

  <!-- - [Overview](#overview)
  - [Getting Started](#getting-started)
  - [Model List](#model-list)
  - [Use SimCSE with Huggingface](#use-simcse-with-huggingface)
  - [Train SimCSE](#train-simcse)
    - [Requirements](#requirements)
    - [Evaluation](#evaluation)
    - [Training](#training)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [SimCSE Elsewhere](#simcse-elsewhere) -->

## Preparation

cd SentEval/data/downstream/
bash download_dataset.sh

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
