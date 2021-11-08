import ast
import unicodedata
from tqdm import tqdm
import re
import json
import sys
import os
from sklearn.datasets import fetch_20newsgroups
import csv


sys.path.append(os.path.abspath("/home/fmg/nishikawa/EASE"))
from utils.sentence_tokenizer import MultilingualSentenceTokenizer

WHITESPACE_REGEXP = re.compile(r'\s+')

class MLDocParser():
    def __call__(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label, sentence = line.strip().split('\t')
                # clean the sentence
                sentence = re.sub(r'\u3000+', '\u3000', sentence)
                sentence = re.sub(r' +', ' ', sentence)
                sentence = re.sub(r'\(c\) Reuters Limited \d\d\d\d', '', sentence)
#                 sentence = normalize_text(sentence)
                yield sentence, label
                
def normalize_text(text):
    text = text.lower()
    text = re.sub(WHITESPACE_REGEXP, ' ', text)
    # remove accents: https://stackoverflow.com/a/518232
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = unicodedata.normalize('NFC', text)
    return text
                

def dataset_load(key):

    key_to_data_path = {
        "Tweet": "data/tweet.txt",
        'Bio': "data/biomedical.txt",
        'SO': "data/stackoverflow.txt",
        'SS': "data/searchsnippets.txt",
        'AG': "data/agnews.txt",
        'G-TS': "data/ts.txt",
        'G-T': "data/t.txt",
        'G-S': "data/s.txt",
        "MD-en": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-fr": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-de": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-ja": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-zh": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-it": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-ru": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-es": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-FS-en": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-FS-fr": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-FS-de": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-FS-ja": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-FS-zh": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-FS-it": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-FS-ru": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "MD-FS-es": "/home/fmg/nishikawa/corpus/mldoc_outputs",
        "WN-T-en": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-ar": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-ja": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-es": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-tr": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-it": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-ko": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-pt": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-uk": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-cs": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-pl": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-ca": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-fi": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-fa": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-nl": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-hu": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-eo": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-T-ru": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-en": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-ar": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-ja": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-es": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-tr": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-it": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-ko": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-pt": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-uk": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-cs": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-pl": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-ca": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-fi": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-fa": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-nl": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-hu": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-eo": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-S-ru": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-en": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-ar": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-ja": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-es": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-tr": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-it": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-ko": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-pt": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-uk": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-cs": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-pl": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-ca": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-fi": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-fa": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-nl": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-hu": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-eo": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-TS-ru": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-en": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-ar": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-ja": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-es": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-tr": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-it": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-ko": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-pt": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-uk": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-cs": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-pl": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-ca": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-fi": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-fa": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-nl": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-hu": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-eo": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "WN-ru": "/home/fmg/nishikawa/EASE/text-clustering/data/wikinews",
        "NC-en": "/home/fmg/nishikawa/news-clustering/dataset/dataset.test.json",
        "NC-de": "/home/fmg/nishikawa/news-clustering/dataset/dataset.test.json",
        "NC-es": "/home/fmg/nishikawa/news-clustering/dataset/dataset.test.json",
        "R8": "/home/fmg/nishikawa/corpus/text_classification/r8-test-stemmed.csv",
        "R52": "/home/fmg/nishikawa/corpus/text_classification/r52-test-stemmed.csv",
        "OH": "/home/fmg/nishikawa/corpus/text_classification/oh-test-stemmed.csv",

    }
    
    key_to_label_path = {
        'Bio': "data/biomedical_label.txt",
        'SO': "data/stackoverflow_label.txt",
        'SS': "data/searchsnippets_label.txt",
    }

    if key in key_to_data_path:
        data_path = key_to_data_path[key]
    
    if key in key_to_label_path:
        label_path = key_to_label_path[key]
        

    if key == "Tweet":
        with open(data_path) as f:
            l_strip = [s.strip() for s in f.readlines()]
        sentences = [ast.literal_eval(d)["text"] for d in l_strip]
        labels = [ast.literal_eval(d)["cluster"] for d in l_strip]
        
    elif key in ["Bio", 'SO', 'SS']:
        with open(data_path) as f:
            sentences = [s.strip() for s in f.readlines()]
        with open(label_path) as f:
            labels = [int(s.strip()) for s in f.readlines()]
            
    elif key in ['AG', 'G-TS', 'G-T', 'G-S']:
        with open(data_path) as f:
            l_strip = [s.strip() for s in f.readlines()]
        sentences = [d.split("\t")[1] for d in l_strip]
        labels = [int(d.split("\t")[0]) for d in l_strip]
        
    elif key.startswith("MD"):
        lang = key[-2:]
        lcode_to_lang = {"en" : "english","fr":"french" ,"de":"german","ja":"japanese", "zh":"chinese", "it": "italian", "ru":"russian", "es":"spanish"}
        lang = lcode_to_lang[lang]
        categories = ["CCAT","MCAT","ECAT","GCAT"]
        categories_index = {t: i for i, t in enumerate(categories)}
        parser = MLDocParser()
        file_path = f"{data_path}/{lang}.test"
        sentence_labels = [(sentence, categories_index[label]) for sentence, label in tqdm(parser(file_path))]  
        sentences = [sentence for sentence, label in sentence_labels]
        if key.startswith("MD-FS"):
            print("set tokenizer...")
            sentence_tokenizer = MultilingualSentenceTokenizer(lang)
            sentences = []
            for sentence, label in sentence_labels:
                try:
                    sentences.append(sentence_tokenizer.tokenize(sentence)[0])
                except Exception as e:
                    print("=== sentence tokenize error ===")
            
        labels = [label for sentence, label in sentence_labels]

    elif key.startswith("WN"):
        lang = key[-2:]
        label_path = data_path + "/" + lang + "_categories.txt"

        if key.startswith("WN-TS"):
            data_path = data_path + "/" + lang + "_title_and_texts.txt"
        elif key.startswith("WN-T"):
            data_path = data_path + "/" + lang + "_titles.txt"
        elif key.startswith("WN-S"):
            data_path = data_path + "/" + lang + "_texts.txt"

        with open(data_path) as f:
            sentences = [s.strip() for s in f.readlines()]

        with open(label_path) as f:
            categories = [s.strip() for s in f.readlines()]
        category_to_idx = {category:idx for idx, category in enumerate(set(categories))}
        labels = [category_to_idx[category] for category in categories]

    elif key.startswith("NC"):
        # https://arxiv.org/pdf/1809.00540.pdf
        lang = key[-2:]
        if lang == "en":
            lang = "eng"
        elif lang == "de":
            lang = "deu"
        elif lang == "es":
            lang = "spa"

        with open(data_path) as f:
            data = json.loads(f.read())

        sentences = [d["text"] for d in data if d["lang"] == lang]
        categories = [d['cluster'] for d in data if d["lang"] == lang]
        category_to_idx = {category:idx for idx, category in enumerate(set(categories))}
        labels = [category_to_idx[category] for category in categories]

    elif key.startswith("20N"):
        newsgroups_test = fetch_20newsgroups(subset='test')
        sentences = newsgroups_test.data
        labels = newsgroups_test.target

    elif key in ['R8', 'R52', 'OH']:
        with open(data_path) as f:
            reader = csv.reader(f)
            data = [row for row in reader]
        data = data[1:]
        sentences = [d[0] for d in data]
        categories = [d[-1] for d in data]
        category_to_idx = {category: idx for idx, category in enumerate(set(categories))}
        labels = [category_to_idx[category] for category in categories]
    return sentences, labels