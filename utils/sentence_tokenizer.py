import en_core_web_sm
import es_core_news_sm
from sentence_splitter import SentenceSplitter
from abc import ABC, ABCMeta, abstractmethod

# 色々なトークナイザをラッパーするクラス
class MultilingualSentenceTokenizer(object):
    def __init__(self, lang):
        self.lang = lang
        self.build_tokenizer()

    def build_tokenizer(self):
        if self.lang in ["en", "es"]:
            self.tokenizer = SpacySentenceTokenizer(self.lang)
        elif self.lang == "ar":
            self.tokenizer = ArabicSentenceTokenizer()
        elif self.lang == "tr":
            self.tokenizer = TurkishSentenceTokenizer()
        else:
            print("対応する言語がありません")

    def tokenize(self, paragraph):
        return self.tokenizer.tokenize(paragraph)


class BaseSentenceTokenizer(metaclass=ABCMeta):
    @abstractmethod
    def tokenize(self, paragraph):
        """
        Returns:
          分割された文群のリストを返す
        """
        pass


class SpacySentenceTokenizer(BaseSentenceTokenizer):

    # コンストラクタ
    def __init__(self, lang):
        super().__init__()
        self.lang = lang
        self.load_nlp()

    def load_nlp(self):
        if self.lang == "en":
            self.nlp = en_core_web_sm.load()
        elif self.lang == "es":
            self.nlp = es_core_news_sm.load()

    def tokenize(self, paragraph):
        return [str(sent) for sent in self.nlp(paragraph).sents]


class ArabicSentenceTokenizer(BaseSentenceTokenizer):
    def tokenize(self, paragraph):
        return arabic_sentence_tokenizer(paragraph)


class TurkishSentenceTokenizer(BaseSentenceTokenizer):
    # コンストラクタ
    def __init__(self):
        self.splitter = SentenceSplitter(language="tr")

    def tokenize(self, paragraph):
        return self.splitter.split(text=paragraph)


def arabic_sentence_tokenizer(paragraph):
    sentences = list()
    temp_sentence = list()
    flag = False
    for ch in paragraph.strip():
        if ch in [u"؟", u"!", u".", u":", u"؛"]:
            flag = True
        elif flag:
            sentences.append("".join(temp_sentence).strip())
            temp_sentence = []
            flag = False

        temp_sentence.append(ch)

    else:
        sentences.append("".join(temp_sentence).strip())
        return sentences
