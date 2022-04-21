from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer
import argparse
import os
from dataset import RawDataLoader
import jsonlist

# please update the version of transformers when you use this code


def upload_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path", type=str)
    parser.add_argument("save_name", type=str)
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_name_or_path)
    #     if "xlm" in args.model_name_or_path:
    #         tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name_or_path)
    #     else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    model.push_to_hub(args.save_name, use_temp_dir=True)
    tokenizer.push_to_hub(args.save_name, use_temp_dir=True)


if __name__ == "__main__":
    upload_model()
    # upload_dataset()


# def save_dataset():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("dataset_name", type=str)
#     parser.add_argument("output_path", type=str, default="uploads/ease_dataset.json")
#     parser.add_argument("--langs", type=str, nargs='+')
#     args = parser.parse_args()

#     if args.dataset_name == "wikidata_hyperlink_type_hn":
#         dataset = RawDataLoader.load(
#             os.path.abspath(os.getcwd()),
#             args.dataset_name,
#             sample_num=1000,
#             hard_negative_num=1,
#             langs=args.langs,
#             min_length=1,
#             seed=42
#         )
#         jsonlist.dump_file(dataset, args.output_path)


# def upload_dataset():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("dataset_name", type=str)
#     parser.add_argument("save_name", type=str)
#     parser.add_argument("--langs", type=str, nargs='+')
#     args = parser.parse_args()

#     if args.dataset_name == "wikidata_hyperlink_type_hn":
#         dataset = RawDataLoader.load(
#             os.path.abspath(os.getcwd()),
#             args.dataset_name,
#             sample_num=1000,
#             hard_negative_num=1,
#             langs=args.langs,
#             min_length=1,
#             seed=42
#         )

#     dataset.push_to_hub(args.save_name)
