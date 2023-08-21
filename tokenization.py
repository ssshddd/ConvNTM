import re
import tokenize
from typing import List
from collections import defaultdict
import multiprocessing

from tqdm import tqdm

# from pyhanlp import *
import spacy
# import nltk
import os

LANG_CLS = defaultdict(lambda:"SpacyTokenizer")
LANG_CLS.update({
    "zh": "HanLPTokenizer",
    "en": "SpacyTokenizer",
})

SPACY_MODEL = {
    "en": "en_core_web_sm",
    "ja": "ja_core_news_sm"
}


class HanLPTokenizer(object):
    def __init__(self, stopwords=None):
        self.pat = re.compile(r'[0-9!"#$%&\'()*+,-./:;<=>?@—，。：★、￥…【】（）《》？“”‘’！\[\\\]^_`{|}~\u3000]+')
        self.stopwords = stopwords
        print("Using HanLP tokenizer")
        
    def tokenize(self, lines: List[str]) -> List[List[str]]:
        docs = []
        for line in tqdm(lines):
            tokens = [t.word for t in HanLP.segment(line)]
            tokens = [re.sub(self.pat, r'', t).strip() for t in tokens]
            tokens = [t for t in tokens if t != '']
            if self.stopwords is not None:
                tokens = [t for t in tokens if not (t in self.stopwords)]
            docs.append(tokens)
        return docs
        
        
class SpacyTokenizer(object):
    def __init__(self, lang="en", stopwords=None):
        self.stopwords = stopwords
        self.pat = re.compile(r'[0-9!"#$%&\'()*+,-.:;<=>?@—，。：★、￥…【】£¥°（）《》？“”‘’！\[\\\]^_`{|}~\u3000]+')
        self.nlp = spacy.load(SPACY_MODEL[lang], disable=['ner', 'parser'])
        if self.stopwords:
            self.nlp.Defaults.stop_words |= self.stopwords
        # print(sorted(list(self.nlp.Defaults.stop_words)), len(self.nlp.Defaults.stop_words))
        print("Using SpaCy tokenizer")
        
    def tokenize(self, lines: List[str]) -> List[List[str]]:
        # for line in lines:
        lines = [line.lower().replace(" ’ ","'").replace(" ‘ ","'").strip() for line in lines]
        lines = [line.replace("n't", " not") for line in lines]
        lines = [line.replace("__eou__", "/sep") for line in lines]
        lines = [re.sub(self.pat, r'', line).strip() for line in lines]
        # print(lines)
        docs = tqdm(self.nlp.pipe(lines, batch_size=1000, n_process=multiprocessing.cpu_count()))
        if self.stopwords:
            docs = [[token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.is_digit or token.is_space)] for doc in docs]
        else:
            docs = [[token.lemma_ for token in doc] for doc in docs]
        return docs


class NLTKTokenizer(object):
    def __init__(self, stopwords=None):
        self.stopwords = stopwords
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        print("Using NLTK tokenizer")

        
    def tokenize(self, lines: List[str]) -> List[List[str]]:
        # for line in lines:
        lines = [line.lower().replace(" ’ ","'").replace(" ‘ ","'").strip() for line in lines]
        lines = [line.replace("n't", " not") for line in lines]
        print(lines)
        convs = []
        for line in lines:
            tokens = self.tokenizer.tokenize(line)
            convs.append(tokens)
        return convs

        # docs = self.nlp.pipe(lines, batch_size=1000, n_process=multiprocessing.cpu_count())
        # docs = [[token.lemma_ for token in doc if not (token.is_stop or token.is_punct)] for doc in docs]
        # return docs


if __name__ == '__main__':
    stopwords = set([l.strip('\n').strip() for l in open(os.path.join('data','stopwords_gensim.txt'),'r',encoding='utf-8')])
    tokenizer=SpacyTokenizer(stopwords=stopwords)
    # tokenizer = NLTKTokenizer()
    text = "Yes , I don ‘ t love to . it ’ s interesting to see who is considered the best in their field."
    print(tokenizer.tokenize([text]))
