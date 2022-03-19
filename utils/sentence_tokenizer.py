# import en_core_web_sm
# # import es_core_news_sm
# import ja_core_news_md
# import de_core_news_sm
# # import ja_core_news_sm
# import zh_core_web_sm
# from sentence_splitter import SentenceSplitter
# from mosestokenizer import MosesSentenceSplitter
from abc import ABC, ABCMeta, abstractmethod

# 色々なトークナイザをラッパーするクラス
class MultilingualSentenceTokenizer(object):
    def __init__(self, lang):
        self.lang = lang
        self.build_tokenizer()

    def build_tokenizer(self):
        # todo mosesに変更
        # zhとjaはそれぞれ
        # if self.lang in ["en"]:
        # if self.lang in ["ja", "zh"]:
        # self.tokenizer = SpacySentenceTokenizer(self.lang)
        # self.tokenizer = MosesSentenceTokenizer(self.lang)
        self.tokenizer = PolyglotSentenceTokenizer(self.lang)

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

# class MosesSentenceTokenizer(BaseSentenceTokenizer):
#     # コンストラクタ
#     def __init__(self, lang):
#         super().__init__()
#         self.lang = lang
#         self.splitter = MosesSentenceSplitter(lang)

#     def tokenize(self, paragraph):
#         return self.splitter([paragraph])


# class SpacySentenceTokenizer(BaseSentenceTokenizer):

#     # コンストラクタ
#     def __init__(self, lang):
#         super().__init__()
#         self.lang = lang
#         self.load_nlp()

#     def load_nlp(self):
#         if self.lang == "ja":
#             # self.nlp = ja_core_news_sm.oad()
#             self.nlp = ja_core_news_md.load()
#         elif self.lang == "zh":
#             self.nlp = zh_core_web_sm.load()
#         elif self.lang == "en":
#             self.nlp = en_core_web_sm.load()
#         elif self.lang == "de":
#             self.nlp = de_core_news_sm.load()
#         # elif self.lang == "es":
#         #     self.nlp = es_core_news_sm.load()

#     def tokenize(self, paragraph):
#         return [str(sent) for sent in self.nlp(paragraph).sents]


# class ArabicSentenceTokenizer(BaseSentenceTokenizer):
#     def tokenize(self, paragraph):
#         return arabic_sentence_tokenizer(paragraph)


# class TurkishSentenceTokenizer(BaseSentenceTokenizer):
#     # コンストラクタ
#     def __init__(self):
#         self.splitter = SentenceSplitter(language="tr")

#     def tokenize(self, paragraph):
#         return self.splitter.split(text=paragraph)


# def arabic_sentence_tokenizer(paragraph):
#     sentences = list()
#     temp_sentence = list()
#     flag = False
#     for ch in paragraph.strip():
#         if ch in [u"؟", u"!", u".", u":", u"؛"]:
#             flag = True
#         elif flag:
#             sentences.append("".join(temp_sentence).strip())
#             temp_sentence = []
#             flag = False

#         temp_sentence.append(ch)

#     else:
#         sentences.append("".join(temp_sentence).strip())
#         return sentences

import polyglot
from polyglot.text import Text
from polyglot.detect.base import logger as polyglot_logger
polyglot_logger.setLevel("ERROR")

class PolyglotSentenceTokenizer(BaseSentenceTokenizer):

    # コンストラクタ
    def __init__(self, lang):
        super().__init__()
        self.lang = lang

    def tokenize(self, paragraph):
        return [sent.raw for sent in Text(paragraph).sentences]
        # return [sent.raw for sent in Text(paragraph, hint_language_code=self.lang).sentences]