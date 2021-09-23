# エンティティの型の基づくnegative sampleつきハイパーリンクwikipediaEACSEデータセットを構築するコード

import os
import sys
from wikipedia2vec.dump_db import DumpDB
from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec
import time
from multiprocessing import Pool
from multiprocessing import Process, Manager
import random
import argparse
import gc


sys.path.append(os.path.abspath(".."))

from utils.utils import pickle_dump, pickle_load, save_model
from utils.interwiki_db import InterwikiDB
from utils.sentence_tokenizer import MultilingualSentenceTokenizer


def get_title_to_same_type_entities(
    title_to_same_type_entities, en_title, en_title_to_rdf_ids, rdf_id_to_en_titles
):
    # title_to_same_type_entities, en_title, en_title_to_rdf_ids, rdf_id_to_en_titles = input_data
    if en_title in en_title_to_rdf_ids and len(en_title_to_rdf_ids[en_title]) > 0:
        results = rdf_id_to_en_titles[random.choice(en_title_to_rdf_ids[en_title])]
        results = random.sample(results, min(len(results), 10))
        results = set(results) - set([en_title])
    else:
        results = set([])
    title_to_same_type_entities[en_title] = results

def process(data):
    (
        entity_per_sentence,
        sentence_tokenizer,
        paragraph,
        wiki_db,
        entity_vocab,
        origin_lang_title,
        title_to_hyperlink_entities,
        title_to_same_type_entities,
        lang_title_to_en_title,
        lang,
    ) = data
    sentence_list = sentence_tokenizer.tokenize(paragraph.text)
    idx = 0
    char_num = len(sentence_list[idx])

    for data in paragraph.wiki_links:
        lang_title = wiki_db.resolve_redirect(data.title)
        en_title = lang_title

        if lang != "en":
            if lang_title not in lang_title_to_en_title:
                continue
            en_title = lang_title_to_en_title[lang_title]

        if en_title not in entity_vocab:
            continue

        start, end = data.span

        # # 該当エンティティと類似度が高いエンティティ
        # en title to same type en titles
        same_type_entities = title_to_same_type_entities[en_title]

        # # 同じ文書内に現れるエンティティ
        # lang title to hyperlink en titles
        hyperlink_entities = title_to_hyperlink_entities[origin_lang_title]

        # ネガティブサンプルの候補エンティティ
        hard_negative_entities = list(same_type_entities - hyperlink_entities)

        while start > char_num and idx + 1 < len(sentence_list):
            idx += 1
            char_num += len(sentence_list[idx])

        data_idx = len(entity_per_sentence)
        entity_per_sentence[data_idx] = {
            "positive_entity": en_title,
            "negative_entity": hard_negative_entities,
            "text": sentence_list[idx],
            "origin_entity": origin_lang_title,
        }


def main():

    parser = argparse.ArgumentParser(description="このプログラムの説明（なくてもよい）")  # 2. パーサを作る
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument('--abstract', action='store_true')
    args = parser.parse_args()
    p = Pool(16)

    print("load files...")
    entity_vocab = pickle_load(
        "/home/fmg/nishikawa/EASE/data/en_wikidata_title_vocab.pkl"
    )
    en_title_to_rdf_ids = pickle_load(
        "/home/fmg/nishikawa/EASE/data/en_title_to_rdf_ids.pkl"
    )
    rdf_id_to_en_titles = pickle_load(
        "/home/fmg/nishikawa/EASE/data/rdf_id_to_en_titles.pkl"
    )

    random.seed(42)
    db_file_path = f"/home/fmg/nishikawa/shinra_data/shinra/entity_db/{args.lang}.db"
    wiki_db = DumpDB(db_file_path)

    print("set tokenizer...")
    sentence_tokenizer = MultilingualSentenceTokenizer(args.lang)

    # 事前にlang entity -> en wikidata entity辞書を作成しておく
    # todo 多言語化
    # print("build lang entity -> en wikidata entity dict")

    lang_title_to_en_title = dict()
    en_title_to_lang_title = dict()

    if args.lang != "en":
        in_file_path = f"/home/fmg/nishikawa/shinra_data/interwiki_db_31.pkl"
        interwiki = InterwikiDB.load(in_file_path)

        for en_title, idx in tqdm(entity_vocab.items()):
            results = interwiki.query(en_title, "en")
            for ans in results:
                if ans[1] == args.lang:
                    lang_title = ans[0]
                    lang_title_to_en_title[lang_title] = en_title
                    en_title_to_lang_title[en_title] = lang_title
                    continue

    print("set title_to_hyperlink_entities entities...")
    # lang title to en hyperlink en titles
    # en title to same type en titles

    title_to_hyperlink_entities = dict()
    title_to_same_type_entities = dict()

    cnt = 0
    # for en_title, idx in tqdm(entity_vocab.items()):
    #     if idx == 0: continue
    #     lang_title = en_title
    #     if args.lang != "en":
    #         if en_title not in en_title_to_lang_title:
    #             continue
    #         lang_title = en_title_to_lang_title[en_title]

    #     try:
    #         if args.lang == "en":
    #             title_to_hyperlink_entities[lang_title] = set(
    #                 [
    #                     wiki_db.resolve_redirect(link.title)
    #                     for paragraph in wiki_db.get_paragraphs(lang_title)
    #                     for link in paragraph.wiki_links
    #                     if wiki_db.resolve_redirect(link.title) in entity_vocab
    #                 ]
    #             )
    #         else:
    #             title_to_hyperlink_entities[lang_title] = set(
    #                 [
    #                     lang_title_to_en_title[wiki_db.resolve_redirect(link.title)]
    #                     for paragraph in wiki_db.get_paragraphs(lang_title)
    #                     for link in paragraph.wiki_links
    #                     if wiki_db.resolve_redirect(link.title)
    #                     in lang_title_to_en_title
    #                     and lang_title_to_en_title[wiki_db.resolve_redirect(link.title)]
    #                     in entity_vocab
    #                 ]
    #             )

    #     except:
    #         pass

    # pickle_dump(dict(title_to_hyperlink_entities), f"/home/fmg/nishikawa/EASE/data/title_to_hyperlink_entities_{args.lang}.pkl")

    title_to_hyperlink_entities = pickle_load(
        "/home/fmg/nishikawa/EASE/data/title_to_hyperlink_entities_en.pkl"
    )

    print("set title_to_same_type_entities...")

    title_to_same_type_entities = dict()

    # for en_title, idx in entity_vocab.items():
    #     get_title_to_same_type_entities(title_to_same_type_entities, en_title, en_title_to_rdf_ids, rdf_id_to_en_titles)

    # pickle_dump(dict(title_to_same_type_entities), "/home/fmg/nishikawa/EASE/data/title_to_same_type_entities.pkl")

    title_to_same_type_entities = pickle_load(
        "/home/fmg/nishikawa/EASE/data/title_to_same_type_entities.pkl"
    )

    manager = Manager()

    print("set manager dicts...")
    entity_per_sentence = manager.dict()
    paragraphs_with_titles = []
    title_to_hyperlink_entities = manager.dict(title_to_hyperlink_entities)
    title_to_same_type_entities = manager.dict(title_to_same_type_entities)

    entity_vocab = manager.dict(entity_vocab)
    lang_title_to_en_title = manager.dict(lang_title_to_en_title)

    print("preprocessing...")

    cnt = 0

    for en_title, idx in tqdm(entity_vocab.items()):
        if idx == 0:
            continue
        lang_title = en_title
        if args.lang != "en":
            lang_title = en_title_to_lang_title[en_title]

        try:
            paragraphs = [paragraph for paragraph in wiki_db.get_paragraphs(lang_title)]


            if args.abstract:
                paragraphs_with_titles.extend(
                    [(paragraph, lang_title) for paragraph in paragraphs if paragraph.abstract]
                )
            else:
                paragraphs_with_titles.extend(
                    [(paragraph, lang_title) for paragraph in paragraphs]
                )
            cnt += 1
        except:
            pass

    datas = [
        (
            entity_per_sentence,
            sentence_tokenizer,
            paragraph,
            wiki_db,
            entity_vocab,
            origin_lang_title,
            title_to_hyperlink_entities,
            title_to_same_type_entities,
            lang_title_to_en_title,
            args.lang,
        )
        for paragraph, origin_lang_title in paragraphs_with_titles
    ]

    print(len(datas))

    print("processing...")
    start = time.time()
    p.map(process, tqdm(datas))
    end = time.time()
    p.close()

    del (
        wiki_db,
        entity_vocab,
        title_to_hyperlink_entities,
        title_to_same_type_entities,
        lang_title_to_en_title,
        paragraphs_with_titles,
    )

    gc.collect()

    entity_per_sentence = dict(entity_per_sentence)

    print("saving...")
    pickle_dump(
        entity_per_sentence,
        f"../data/wikidata_hyperlinks_with_type_hardnegatives_abst_{args.abstract}_{args.lang}.pkl",
    )


if __name__ == "__main__":
    main()
