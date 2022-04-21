## EASE: Entity-Aware Contrastive Learning of Sentence Embedding

EASE is a novel method for learning sentence embeddings via contrastive learning between sentences and their related entities proposed in out paper EASE: Entity-Aware Contrastive Learning of Sentence Embedding.
This repository contains the source code to train the model and evaluate it with downstream tasks.

<!-- TODO NAACL2022の記述 -->

<img src="figure/ease.png" width="70%">

## Released Models

Our published models are listed as follows.
You can use these models by using [HuggingFace's Transformers](https://github.com/huggingface/transformers).

<!-- huggingface libriryで使えるよ -->


|              **Monolingual Models**             | **Avg. STS** | **Avg. STC** |
|:-------------------------------|:--------:|:--------:|
|  [sosuke/ease-bert-base-uncased](https://huggingface.co/sosuke/ease-bert-base-uncased) |   77.0 |  63.1    |
| [sosuke/ease-roberta-base](https://huggingface.co/sosuke/ease-roberta-base) |  76.8 |  58.6   |
|              **Multilingual Models**              | **Avg. mSTS** | **Avg. mSTC** |
|  [sosuke/ease-bert-base-multilingual-cased](https://huggingface.co/princeton-nlp/sup-simcse-bert-large-uncased)  |   57.2  | 36.0 |
|     [sosuke/ease-xlm-roberta-base](https://huggingface.co/sosuke/ease-xlm-roberta-base)     |   57.1 | 35.4 |


## Installation

<!-- Python 3.7.6 -->

Run the following script to install the dependent libraries.
```bash
pip install -r requirements.txt
```

Before training, please download the datasets for training and evaluation.
```bash
bash download_all.sh
```
<!-- TODO check download_all.sh -->


## Evaluation

We provide evaluation codes for sentence embeddings including Short Text Clustering, Cross-lingual Parallel Matching and Cross-lingual Text Classification.

Set your model or path of tranformers-based checkpoint (`--model_name_or_path`),
pooling method type (`--pooler`), and what set of tasks (`--task_set`).
See the example codes below.

##### Semantic Textual Similarity
```bash
python evaluation.py \
    --model_name_or_path sosuke/ease-bert-base-multilingual-cased \ 
    --pooler avg \ 
    --task_set cl-sts 
```

##### Short Text Clustering
```bash
python text-clustering/evaluate.py \
    --model_name_or_path sosuke/ease-bert-base-multilingual-cased \
    --pooler avg \ 
    --task_set cl
```

##### Cross-lingual Parallel Matching
```bash
python parallel-matching/similar_sentence_search.py \
    --model_name_or_path sosuke/ease-bert-base-multilingual-cased \ 
    --pooler avg 
```

##### Cross-lingual Text Classification
```bash
python cross-lingual-transfer/main.py \
    --model_name_or_path sosuke/ease-bert-base-multilingual-cased \ 
    --pooler avg
```

Please refer to each evaluation code for detailed descriptions of arguments.


## Training


You can train an EASE model in a monolingual setting using English Wikipedia sentences or in a multilingual setting using Wikipedia sentences in 18 languages.

We provide example trainig scripts for both monolingual ([train_monolingual_ease.sh](https://github.com/Sosuke115/EASE/blob/main/train_monolingual_ease.sh)) and multilingual ([train_multilingual_ease.sh](https://github.com/Sosuke115/EASE/blob/main/train_multilingual_ease.sh)) setting.
<!-- TODO link -->



## MewsC-16

We constructed [MewsC-16](https://github.com/Sosuke115/EASE/tree/main/text-clustering/data/mewsc16) (**M**ultilingual Short Text **C**lustering Dataset for N**ews** in **16** languages) from Wikinews.
The dataset contains topic sentences from Wikinews articles in 13 categories and 16 languages. More detailed information is available in our paper, Appendix E.

<!-- TODO link -->
<!-- TODO reproduction code -->

##### Statistics

<table border=0><tr><td> 
<table border>
<tr>
<th>Language </th><th> sentences </th><th> label types</th>
</tr>
<tr  align="right">
<td>ar </td><td>2,224 </td><td>11 </td>
</tr>
<tr  align="right">
<td>ca </td><td>3,310 </td><td>11 </td>
</tr>
<tr  align="right">
<td>cs </td><td>1,534 </td><td>9 </td>
</tr>
<tr  align="right">
<td>de </td><td>6,398 </td><td>8 </td>
</tr>
<tr  align="right">
<td>en </td><td>12,892 </td><td>13 </td>
</tr>
<tr  align="right">
<td>eo </td><td>227 </td><td>8 </td>
</tr>
<tr  align="right">
<td>es </td><td>6,415 </td><td>11 </td>
</tr>
<tr  align="right">
<td>fa </td><td> 773</td><td>9 </td>
</tr>
</table>
</td>
<td valign="top"> 
<table border>
<tr>
<th>Language </th><th> sentences </th><th> label types</th>
</tr>
<tr  align="right">
<td>fr </td><td>10,697 </td><td>13 </td>
</tr>
<tr  align="right">
<td>ja </td><td>1,984 </td><td>12 </td>
</tr>
<tr  align="right">
<td>ko </td><td>344 </td><td>10 </td>
</tr>
<tr  align="right">
<td>pl </td><td>7,247 </td><td>11</td>
</tr>
<tr  align="right">
<td>pt </td><td>8,921 </td><td>11 </td>
</tr>
<tr  align="right">
<td>ru </td><td>1,406 </td><td>12 </td>
</tr>
<tr  align="right">
<td>sv </td><td>584 </td><td>7 </td>
</tr>
<tr  align="right">
<td>tr </td><td> 459</td><td>7 </td>
</tr>

</table>
</td></tr></table> 

## Citation
[TBA]

<!-- TODO -->