# en_wikidata_title_vocab、en_title_to_rdf_ids、rdf_id_to_en_titlesを作成するコード

# wiki_dbに存在するエンティティに絞る
# タスクも古いのでwiki_dbもある程度古くて良さそう
#

import sys
import os
from collections import defaultdict
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec import Wikipedia2Vec

sys.path.append(os.path.abspath(".."))
from utils.utils import pickle_dump, pickle_load, save_model


def main():

    vector_path = "/home/fmg/nishikawa/multilingual_classification_using_language_link/data/enwiki.768.vec"
    embedding = Wikipedia2Vec.load(vector_path)

    en_title_to_wikidata_id = pickle_load(
        "/home/fmg/nishikawa/EASE/data/en_title_to_wikidata_id.pkl"
    )

    wikidata_id_to_rdf_ids = pickle_load(
        "/home/fmg/nishikawa/EASE/data/wikidata_id_to_rdf_ids.pkl"
    )

    en_wikidata_title_vocab = {"[PAD]": 0}
    en_title_to_rdf_ids = {}
    rdf_id_to_en_titles = defaultdict(list)
    en_title_to_link_count = {}
    print("building en_title_to_rdf_ids....")
    cnt = 0

    for data, count in tqdm(zip(embedding.dictionary.entities(), embedding.dictionary._entity_stats[:, 0])):
        en_title = data.title
        if count > 10:
            en_title_to_link_count[en_title] = count
            en_wikidata_title_vocab[en_title] = len(en_wikidata_title_vocab)
            if en_title in en_title_to_wikidata_id:
                rdf_ids = wikidata_id_to_rdf_ids[en_title_to_wikidata_id[en_title]]
                en_title_to_rdf_ids[en_title] = rdf_ids
                for rdf_id in rdf_ids:
                    rdf_id_to_en_titles[rdf_id].append(en_title)
                cnt += 1

    pickle_dump(
        en_title_to_link_count,
        "/home/fmg/nishikawa/EASE/data/en_title_to_link_count.pkl",
    )

    print(cnt)
    pickle_dump(
        en_wikidata_title_vocab,
        "/home/fmg/nishikawa/EASE/data/en_wikidata_title_vocab.pkl",
    )
    pickle_dump(
        en_title_to_rdf_ids, "/home/fmg/nishikawa/EASE/data/en_title_to_rdf_ids.pkl"
    )
    pickle_dump(
        rdf_id_to_en_titles, "/home/fmg/nishikawa/EASE/data/rdf_id_to_en_titles.pkl"
    )


if __name__ == "__main__":
    main()
