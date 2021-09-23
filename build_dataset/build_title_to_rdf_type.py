from collections import defaultdict
from tqdm import tqdm
import bz2
import ujson
import sys
import os

sys.path.append(os.path.abspath(".."))
from utils.utils import pickle_dump, pickle_load, save_model


def main():
    wikidata_id_to_rdf_ids = defaultdict(list)
    en_title_to_wikidata_id = dict()
    wiki_data_file = "/home/fmg/nishikawa/corpus/latest-all.json.bz2"

    with bz2.BZ2File(wiki_data_file) as f:
        for (n, line) in tqdm(enumerate(f)):
            line = line.rstrip().decode("utf-8")
            if line in ("[", "]"):
                continue

            if line[-1] == ",":
                line = line[:-1]
            obj = ujson.loads(line)
            if obj["type"] != "item":
                continue

            wikidata_id = obj["id"]
            try:
                en_title = obj["labels"]["en"]["value"]
                en_title_to_wikidata_id[en_title] = wikidata_id

            except:
                print("###there is not en title###")

            try:
                rdf_wikidata_ids = [
                    data["mainsnak"]["datavalue"]["value"]["id"]
                    for data in obj["claims"]["P31"]
                ]
                wikidata_id_to_rdf_ids[wikidata_id].extend(rdf_wikidata_ids)
            except:
                print("###there is not rdf ids###")

    pickle_dump(
        en_title_to_wikidata_id,
        "/home/fmg/nishikawa/EASE/data/en_title_to_wikidata_id.pkl",
    )
    pickle_dump(
        wikidata_id_to_rdf_ids,
        "/home/fmg/nishikawa/EASE/data/wikidata_id_to_rdf_ids.pkl",
    )


if __name__ == "__main__":
    main()
