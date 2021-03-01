import os
import re
import csv
import sys
from glob import iglob
from pathlib import Path
from glove import Corpus, Glove

sys.path.append('..')
from module.dirHandler import mkdir_p, del_folder

BASE_DIR = "/docker/data/ksb/TestSampleDir"
DATA_BASE_DIR = os.path.join(BASE_DIR, "articles")

ORIGIN_PATH = os.path.join(DATA_BASE_DIR,"Origin-Data")
PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Preprocessed-Data")
PRETTY_PATH = os.path.join(DATA_BASE_DIR,"Pretty-Data")
SUMMARY_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Summary-Preprocessed-Data")
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

    result = []
    forCount = []
    wordDict = {}

    for idx, proc_article_path in enumerate(iglob(os.path.join(PREPROCESSED_PATH, '**.csv'), recursive=False)):
        
        f_proc= open(proc_article_path, 'r', newline="\n", encoding="utf-8")
        for [idx, title, contents] in csv.reader(f_proc):
            if contents is '': continue

            cont_list = contents.split("\t")
            forCount += [token for sent in cont_list for token in sent.split()]
            result += [sent.split() for sent in cont_list]
            list(map(append_to_dict, [token for sent in cont_list for token in sent.split()]))
            
        f_proc.close()

    print("전체 token의 개수 : {len}".format(len=len(forCount)))
    print("중복되지 않은 token의 개수 : {len}".format(len=len(list(set(forCount)))))

    print("Drop less than {}".format(MIN_COUNT))
    rmMinCount ={}
    for key, val in wordDict.items():
        if val < MIN_COUNT : continue
        rmMinCount[key]=val

    
    rmMinCount[START_TOKEN[0]] = rmMinCount[END_TOKEN[0]] = len(result)
    rmMinCount_summary[START_TOKEN[0]] = rmMinCount_summary[END_TOKEN[0]] = len(result)

    result = list(map(lambda content : START_TOKEN + content + END_TOKEN, result))
    summary_result = list(map(lambda content : START_TOKEN + content + END_TOKEN, summary_result))
        
    print("최종 token 개수 : {}".format(len(rmMinCount)))

    print("Train Word-Embedding Model")
    corpus = Corpus() 
    corpus.fit(rmMinCountList, window=5)

    glove = Glove(no_components=D_MODEL, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=10, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)

    mkdir_p(MODEL_PATH)

    corpus_path = os.path.join(MODEL_PATH, "input-corpus-{d_model}-{mincount}.model".format(d_model=D_MODEL, mincount=MIN_COUNT))
    model_path = os.path.join(MODEL_PATH, 'glove-{d_model}-{mincount}.model'.format(d_model=D_MODEL, mincount=MIN_COUNT))

    print("Save Word-Embedding Model")
    glove.save(model_path)
    corpus.save(corpus_path)


