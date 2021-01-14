import re
import nltk
import os
from konlpy.tag import Komoran
from loadNews import mkdir_p, del_folder
from nltk.tokenize import RegexpTokenizer

nltk.download('punkt')

BASE_DIR = "."
ORIGIN_PATH = os.path.join(BASE_DIR,"Origin-Data")
PREPROCESSED_PATH = os.path.join(BASE_DIR,"Preprocessed-Data")
PRETTY_PATH = os.path.join(BASE_DIR,"Pretty-Data")
SWORDS_PATH = os.path.join(BASE_DIR, "StopWordList.txt")

class TextPreprocessor:
    def __init__(self):
        self.retokenize = RegexpTokenizer("[\w]+")
        self.swords = []
        self.tokenizer = {}
        self.tagger = Komoran()

    def cleanContent(self, content, media):
        rmBracket = re.sub('(\([^)]*\)|\[[^]]*\])', '', content)  # 괄호 안 내용 제거
        rmMedia = rmBracket.replace(media, ' ')  # 언론사명 제거
        rmReporter = re.sub('[가-힣]{2,5}\s?기자', ' ', rmMedia) # 기자 이름 제거
        rmSpace = re.sub('\s+', ' ', rmReporter)  # 중복 공백, 탭, 개행 제거
        rmEmail = re.sub('[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*\.[a-zA-Z]{2,3}', ' ', rmSpace) # 이메일 제거

        return rmEmail

    def removeSpecialChar(self, text):
        return ' '.join(self.retokenize.tokenize(text))

    def loadSwords(self, filename):
        self.swords = []
        with open(filename, 'r') as f:
            swords = f.readlines()
            self.swords = [tag for sword in self.swords for tag in self.tagger.pos(sword.strip()) if
                           tag[1] in ('NNG', 'NNP', 'VV', 'VA')]

        self.tokenizer = lambda sent: filter(lambda x: x not in self.swords and x[1] in ('NNG', 'NNP', 'VV', 'VA'),
                                             self.tagger.pos(sent))

        return self.swords

    def removeSwords(self, text):
        return ' '.join([noun for (noun, pos) in list(self.tokenizer(text))])

class Article:
    def __init__(self, articleInfo):
        self.title = articleInfo[0]
        self.media = articleInfo[1]
        self.content = articleInfo[2:]

    def readContent(self):
        for line in self.content:
            if line is '': continue
            yield line


class ArticleReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.rgxSplitter = re.compile('([.!?:](?:["\']|(?![0-9])))')

    def __iter__(self):
        with open(self.filepath, encoding='utf-8') as f:
            title = f.readline()[:-1]
            yield title
            content = f.readline()[:-1]

            media = f.readline()[:-1]
            yield media

            docs = self.rgxSplitter.split(content)
            for s in map(lambda a, b: a + b, docs[::2], docs[1::2]):
                if not s: continue
                yield s

def saveTextFile(baseDir, media, filename, sentences):

    mkdir_p(os.path.join(baseDir, media))
    save_path = os.path.join(os.path.join(baseDir, media), filename)

    with open(save_path, 'w') as f:
        f.write('/n'.join([sentence for sentence in sentences if sentence is not '']))



if __name__ == '__main__':

    del_folder(PREPROCESSED_PATH)
    del_folder(PRETTY_PATH)

    preprocessor = TextPreprocessor()
    preprocessor.loadSwords(SWORDS_PATH)


    media_list = os.listdir(ORIGIN_PATH)

    for media in media_list:

        media_path = os.path.join(ORIGIN_PATH, media)
        article_list = os.listdir(media_path)

        for article_name in article_list:

            reader = ArticleReader(os.path.join(media_path, article_name))
            article = Article(list(filter(None, reader)))

            prettyLine = []
            preprocessedLine = []
            for line in article.readContent():
                cleanLine = preprocessor.cleanContent(line, media)
                cleanLine = preprocessor.removeSpecialChar(cleanLine)

                rmSwordLine = preprocessor.removeSwords(cleanLine)

                preprocessedLine.append(rmSwordLine)

            saveTextFile(PREPROCESSED_PATH, media, article_name, preprocessedLine)
            saveTextFile(PRETTY_PATH, media, article_name, article.readContent())
