from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Komoran
import networkx
import re
from shutil import rmtree
import os
import math

BASE_DIR = "../crawlNews/articles"
PREPROCESSED_PATH = os.path.join(BASE_DIR,"Preprocessed-Data")
PRETTY_PATH = os.path.join(BASE_DIR,"Pretty-Data")
ORIGIN_PATH = os.path.join(BASE_DIR,"Origin-Data")
SUMMARY_PATH = os.path.join(BASE_DIR,"Summary-Data")
SWORDS_PATH = os.path.join(BASE_DIR, "StopWordList.txt")


class RawTextReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.rgxSplitter = re.compile("/n")

    def __iter__(self):
        for line in open(self.filepath, encoding='utf-8'):
            ch = self.rgxSplitter.split(line)
            for s in ch:
                yield s


class Document:
    def __init__(self, originSentenceIter, procSentenceIter):
        self.originSents = list(filter(None, originSentenceIter))
        self.procSents = list(filter(None, procSentenceIter))

    def getOriginSet(self):
        return self.originSents

    def getSentsZip(self):
        return zip(self.originSents, self.procSents)


class TextRank:
    def __init__(self, **kargs):
        self.graph = None
        self.coef = kargs.get('coef', 1.0)
        self.threshold = kargs.get('threshold', 0.005)
        self.dictCount = {}
        self.dictBiCount = {}

        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = {}

    def loadSents(self, document, tokenizer):

        sentSet = []
        for origin, proc in document.getSentsZip():
            tagged = set(filter(None, tokenizer(proc)))
            if len(tagged) < 2: continue
            self.dictCount[len(self.dictCount)] = origin
            sentSet.append(tagged)

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(document.getOriginSet())

        sents_distances = (self.tfidf_matrix * self.tfidf_matrix.T)
        sents_distances_matrix = sents_distances.toarray()

        for i in range(len(self.dictCount)):
            for j in range(i + 1, len(self.dictCount)):
                similarity = sents_distances_matrix[i, j]

                if similarity < self.threshold: continue
                self.dictBiCount[i, j] = similarity

    def build(self):
        self.graph = networkx.Graph()
        self.graph.add_nodes_from(self.dictCount.keys())
        for (a, b), n in self.dictBiCount.items():
            self.graph.add_edge(a, b, weight=n * self.coef + (1 - self.coef))

    def rank(self):
        return networkx.pagerank(self.graph, weight='weight')

    def summarize(self, ratio=0.333):
        r = self.rank()
        ks = sorted(r, key=r.get, reverse=True)[:int(len(r) * ratio)]
        return ' '.join(map(lambda k: self.dictCount[k], sorted(ks)))


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

def saveTextFile(baseDir, media, filename, sentences):

    mkdir_p(os.path.join(baseDir, media))
    save_path = os.path.join(os.path.join(baseDir, media), filename)

    with open(save_path, 'w') as f:
        f.write('/n'.join([sentence for sentence in sentences if sentence is not '']))


if __name__ == '__main__':

    del_folder(SUMMARY_PATH)
    media_list = os.listdir(PREPROCESSED_PATH)

    for media in media_list :

        origin_article_list = os.listdir(os.path.join(PRETTY_PATH, media))
        proc_article_list = os.listdir(os.path.join(PREPROCESSED_PATH, media))

        for article in origin_article_list :

            origin_article_path = os.path.join(os.path.join(PRETTY_PATH, media), article)
            proc_article_path = os.path.join(os.path.join(PREPROCESSED_PATH, media), article)

            tr = TextRank()

            tagger = Komoran()
            tr.loadSents(Document(RawTextReader(origin_article_path), RawTextReader(proc_article_path)),
                         lambda sent: filter(lambda x: x[1] in ('NNG', 'NNP', 'VV', 'VA'),
                                             tagger.pos(sent)))
            tr.build()

            ranks = tr.rank()
            for k in sorted(ranks, key=ranks.get, reverse=True)[:100]:
                print("\t".join([str(k), str(ranks[k]), str(tr.dictCount[k])]))

            summary = tr.summarize(0.2)
            print(summary)

            saveTextFile(SUMMARY_PATH, media, article, summary)





