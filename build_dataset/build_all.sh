# build en_title_to_wikidata_id and wikidata_id_to_rdf_ids from latest-all.json.bz2

OUTPUT_DIR="../data"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir $OUTPUT_DIR
fi
LANG="en"
WIKIDATA_DUMP_PATH="${OUTPUT_DIR}/latest-all.json.bz2"
WIKIPEDIA2VEC_PATH="${OUTPUT_DIR}/data/enwiki.fp16.768.vec"
DUMP_DB_PATH="${OUTPUT_DIR}/data/en.db"
INTERWIKI_DB_PATH="${OUTPUT_DIR}/data/interwiki_db_144.pkl"
EASE_DATASET_PATH="${OUTPUT_DIR}/ease_dataset_en.pkl"

python build_title_to_rdf_type.py $WIKIDATA_DUMP_PATH $OUTPUT_DIR --test_mode
python build_en_title_rdf_dict.py $WIKIPEDIA2VEC_PATH $OUTPUT_DIR
python build_type_hardnegative_dataset.py $DUMP_DB_PATH $INTERWIKI_DB_PATH $OUTPUT_DIR --test_mode
python dataset_sampler.py $LANG $EASE_DATASET_PATH $OUTPUT_DIR