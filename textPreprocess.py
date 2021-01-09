import re
import nltk
import os
from loadNews import mkdir_p
from konlpy.tag import Okt
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from shutil import rmtree

nltk.download('punkt')

#BASE_DIR = "/content/gdrive/My Drive/Colab Notebooks/Text-preprocessing-Data/"
BASE_DIR = "."
ARTICLE_MEDIA_PATH = os.path.join(BASE_DIR,"articles")
TARGET_PATH = os.path.join(BASE_DIR,"preprocessed")
SWORDS_FILE_PATH = os.path.join(BASE_DIR, "StopWordList.txt")

def readArticle(filename):

    f = open(filename, 'r', encoding='utf-8')
    title = f.readline()[:-1]
    content = f.readline()[:-1]
    media = f.readline()[:-1]
    f.close()

    return title, media, content

def cleanContent(content, media):
    content = re.sub('\s+', ' ', content)  # 중복 공백, 탭, 개행 제거
    content = re.sub(r'\([^)]*\)', '', content)  # 괄호 안 숫자 제거
    content = content.replace(media, '')  # 언론사명 제거

    return content

def removeSpecialChar(text):
    retokenize = RegexpTokenizer("[\w]+")
    return ' '.join(retokenize.tokenize(text))

def getStopWord(swords_filename):
    swords = []
    with open(swords_filename, 'r') as f:
        swords = f.readlines()
        swords = [sword.strip() for sword in swords]

    return swords

def delStopWord(sentence):
    if sentence is '':
        return None

    okt = Okt()
    swords = getStopWord(SWORDS_FILE_PATH)
    return ' '.join([word for word in okt.morphs(sentence) if word not in swords])

def getRmSwordSentences(sentences):
    rmSwordSentences = []
    for sentence in sentences:
        sentence = delStopWord(sentence)
        if sentence is not None : rmSwordSentences.append(sentence)

    return rmSwordSentences

def savePreprocessedText(media, article, nouns):

    mkdir_p(os.path.join(TARGET_PATH, media))
    save_path = os.path.join(os.path.join(TARGET_PATH, media), article)

    with open(save_path, 'w') as f:
        f.write(title)
        preprocessed = ""
        for noun in nouns:
            preprocessed += noun + "/"
        f.write(preprocessed)


if __name__ == '__main__':
    media_list = os.listdir(ARTICLE_MEDIA_PATH)

    for media in media_list:

        media_path = os.path.join(ARTICLE_MEDIA_PATH, media)
        article_list = os.listdir(media_path)

        for article in article_list:
            title, media, content = readArticle(os.path.join(media_path, article))
            content = cleanContent(content, media)

            sentences = sent_tokenize(content)
            sentences = [removeSpecialChar(sentence) for sentence in sentences]

            rmSwordSentences = getRmSwordSentences(sentences)

            savePreprocessedText(media, article, rmSwordSentences)
