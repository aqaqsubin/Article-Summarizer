from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import os
import re
import nltk
import os
from konlpy.tag import Okt
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from shutil import rmtree

nltk.download('punkt')

BASE_DIR = "articles"
ARTICLE_MEDIA_PATH = os.path.join(BASE_DIR,"Origin-Data")
TARGET_PATH = os.path.join(BASE_DIR,"Preprocessed-Data")
SWORDS_FILE_PATH = os.path.join(BASE_DIR, "StopWordList.txt")


def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def del_folder(path):
    try:
        rmtree(path)
    except:
        pass


def text2Sentences(text):
    return text.split('/')


def sentences2Text(sentences):
    return '/'.join([sentence for sentence in sentences])


def readArticle(filename):
    f = open(filename, 'r', encoding='utf-8')
    # title = f.readline()
    text = f.readline()
    f.close()

    return text


def getStopWord(swords_filename):
    swords = []
    with open(swords_filename, 'r') as f:
        swords = f.readlines()
        swords = [sword.strip() for sword in swords]

    return swords


def getNouns(sentences):
    okt = Okt()
    swords = getStopWord(SWORDS_FILE_PATH)

    nouns = []
    for sentence in sentences:
        if sentence is not '':
            nouns.append(' '.join([noun for noun in okt.morphs(sentence) if noun not in swords and len(noun) > 1]))

    return nouns


class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []

    def build_sent_graph(self, sentences):
        tfidf_mat = self.tfidf.fit_transform(sentences).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return self.graph_sentence

    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)

        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word]: word for word in vocab}


class Rank(object):
    def get_ranks(self, graph, d=0.85):
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0
            link_sum = np.sum(A[:, id])
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
        B = (1 - d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)
        return {idx: r[0] for idx, r in enumerate(ranks)}


class TextRank(object):
    def __init__(self, text):
        self.sentences = text2Sentences(text)
        self.nouns = getNouns(self.sentences)

        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)

        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)

        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx)

    def summarize(self, sent_num=3):
        summary = []
        index = []
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        index.sort()

        for idx in index:
            summary.append(self.sentences[idx])

        return summary

    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)

        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)

        keywords = []
        index = []
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)

        for idx in index:
            keywords.append(self.idx2word[idx])
        return keywords


if __name__ == '__main__':

    media_list = os.listdir(TARGET_PATH)
    media_path = os.path.join(TARGET_PATH, media_list[0])

    document = os.listdir(media_path)[0]
    text = readArticle(os.path.join(media_path, document))

    print(text)

    textrank = TextRank(text)
    for row in textrank.summarize(3):
        print(row + '\n')
