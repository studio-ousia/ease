import numpy as np
import re
from tqdm import tqdm


class GloveSentenceEncoder:
    def __init__(self, glove_vector_path):
        self.glove_embedding = self.read_embedding(glove_vector_path)
        self.dim = self.glove_embedding["apple"].shape[0]

    def read_embedding(self, path):
        glove_embedding = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                glove_embedding[word] = coefs
        return glove_embedding

    def text_to_words(self, text):
        text = re.sub(r"[^a-zA-Z]", " ", text)
        words = text.lower().split()
        return words

    def encode_sentence(self, text):
        words = self.text_to_words(text)
        embeddings = [
            self.glove_embedding[w] for w in words if w in self.glove_embedding
        ]
        if len(embeddings) == 0:
            return np.zeros(self.dim)
        return np.mean(embeddings, axis=0)

    def encode_sentences(self, sentences):
        print("encoding..")
        return np.array(
            [list(self.encode_sentence(sentence)) for sentence in tqdm(sentences)]
        )
