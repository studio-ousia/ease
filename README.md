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

## Evaluation

### Semantic textual similarity

### Short text clustering

```bash
python text-clustering/evaluate.py \
    --model_name_or_path bert-base-uncased \ # set your model path
    --pooler avg \ # set pooling method
    --task_set mono \ # mono or cl or full
```

### Cross-lingual Parallel Matching

### Cross-lingual Zero-shot Classification
