import argparse
import re
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="/home/nchaso/EASE/downstreams/text-clustering/data/mewsc16/ja_sentences.txt",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/nchaso/EASE/downstreams/text-clustering/data/mewsc16/cleaned_ja_sentences.txt",
    )
    args = parser.parse_args()

    regex = re.compile(
        r"thumb\||right\||left\||upright=1\||náhled\||eta\||frame\||\d\d\dpx\||upright\||слева\||\d\d\dпкс\||"
    )

    with open(args.input_path) as f:
        texts = [regex.sub("", s.strip()) for s in tqdm(f.readlines())]

    with open(args.output_path, mode="w") as f:
        f.write("\n".join(texts))


if __name__ == "__main__":
    main()
