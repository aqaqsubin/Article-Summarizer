import nltk
import os
import re
import csv
import sys
import pandas as pd
from glob import iglob
from pathlib import Path
from gensim.models import Word2Vec

sys.path.append('..')
from module.dirHandler import mkdir_p, del_folder

nltk.download('punkt')

BASE_DIR = "/data/ksb/TestSampleDir"
DATA_BASE_DIR = os.path.join(BASE_DIR, "articles")

ORIGIN_PATH = os.path.join(DATA_BASE_DIR,"Origin-Data")
PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Preprocessed-Data")
PRETTY_PATH = os.path.join(DATA_BASE_DIR,"Pretty-Data")
SWORDS_PATH = os.path.join(DATA_BASE_DIR, "StopWordList.txt")
MODEL_PATH = os.path.join(os.path.join(Path(os.getcwd()).parent, "Word-Embedding-Model"))

MIN_COUNT = 10
D_MODEL = 128
START_TOKEN = ['<SOS>']
END_TOKEN = ['<EOS>']

def append_to_dict(word):
    
    if word in wordDict:
        wordDict[word] += 1
    else :
        wordDict[word] = 1
        
    return wordDict


if __name__ == '__main__':
    media_list = os.listdir(PREPROCESSED_PATH)

    wordDict = {}
    for idx, proc_article_path in enumerate(iglob(os.path.join(PREPROCESSED_PATH, '**.csv'), recursive=False)):
        
        f_proc= open(proc_article_path, 'r', newline="\n", encoding="utf-8")
        for [idx, title, contents] in csv.reader(f_proc):
            if contents is '': continue

            cont_list = contents.split("\t")
            list(map(append_to_dict, [token for sent in cont_list for token in sent.split()]))

    rmMinCount ={}
    for key, val in wordDict.items():
        if val < 3 : continue
        rmMinCount[key]=val
    
    rmMinCount[START_TOKEN[0]] = rmMinCount[END_TOKEN[0]] = len(result)
    result = list(map(lambda content : START_TOKEN + content + END_TOKEN, result))

    rmMinCountList = list(map(lambda content : [token for token in content if token in rmMinCount], result))

    model = Word2Vec(sentences=rmMinCountList, size=D_MODEL, window=5, min_count=MIN_COUNT, workers=4, sg=1, iter=10)
    mkdir_p(MODEL_PATH)

    wordvec_path = os.path.join(MODEL_PATH, "word2vec-{}-{}.wordvectors".format(D_MODEL, MIN_COUNT))
    model_path = os.path.join(MODEL_PATH, 'word2vec-{}-{}.model'.format(D_MODEL, MIN_COUNT))

    word_vectors = model.wv
    word_vectors.save(wordvec_path)
    model.save(model_path)