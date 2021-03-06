# _*_ coding: utf-8 _*_

import re
import nltk
import os
import csv
import sys
import math
import argparse
import pandas as pd
import numpy as np
from glob import iglob
from functools import reduce
from konlpy.tag import Komoran
import sentencepiece as spm

from gensim.summarization.summarizer import summarize

nltk.download('punkt')

from module.dirHandler import mkdir_p, del_folder
from module.articleHandler import Article
from module.textPreprocessor import TextPreprocessor
from module.parse import ParseBoolean

MAX_COUNT = 300
MIN_COUNT = 10
TITLE_MIN_COUNT = 3

BASE_DIR =os.getcwd()
DATA_BASE_DIR = os.path.join(BASE_DIR, 'articles')
SRC_BASE_DIR = os.path.join(BASE_DIR, 'src')

ORIGIN_PATH = os.path.join(DATA_BASE_DIR,"Origin-Data")

PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Preprocessed-Data")
TITLE_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Title-Preprocessed-Data")

SUMMARY_PATH = os.path.join(DATA_BASE_DIR,"Summary-Data")
SUMMARY_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Summary-Preprocessed-Data")

VALID_PATH = os.path.join(DATA_BASE_DIR,"Valid-Data")
VAL_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Valid-Preprocessed-Data")
VAL_SUMMARY_PATH = os.path.join(DATA_BASE_DIR,"Valid-Summary-Data")
VAL_SUMMARY_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Valid-Summary-Preprocessed-Data")
VAL_TITLE_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Valid-Title-Preprocessed-Data")

SWORDS_PATH = os.path.join(DATA_BASE_DIR, "StopWordList.txt")
WORD_ENCODING_DIR = os.path.join(SRC_BASE_DIR, 'Word-Encoding-Model')

# Get argument to determine whether to load a word model
parser = argparse.ArgumentParser(description="Description")
parser.add_argument('--tokenizer', required=True, type=ParseBoolean, help="If True, use sentencepiece tokenizer")
args = parser.parse_args()

tokenizer = args.tokenizer

# If tokenizer is true, load word encoding model
if tokenizer:
    sp_src = spm.SentencePieceProcessor()
    model_num = len(list(iglob(os.path.join(WORD_ENCODING_DIR, 'spm-input-*.vocab'), recursive=False))) -1
    sp_src.Load(os.path.join(WORD_ENCODING_DIR, 'spm-input-{}.model').format(model_num))

    print("Selected Model Num : {}".format(model_num))

get_line_token_count = lambda sent : len(sp_src.encode_as_ids(sent)) if tokenizer else len(sent.split())
get_token_count = lambda sents : reduce(lambda x, y : x + y, map(get_line_token_count, sents))

def split_by_max_token(sents):
    
    """Returns length of sentence list and sentence list
    Args:
        sents (list): List of sentences before being refined
    Returns:
        len(sents) : length of Sentence list splitted by MAX COUNT 
        sents : Sentence list splitted by MAX COUNT 
    """
    token_lens = [len(sp_src.encode_as_ids(sent)) if tokenizer else len(sent.split()) for sent in sents]
    total_sum = 0

    for idx, length in enumerate(token_lens):
        if total_sum + length > MAX_COUNT : 
            return idx, sents[:idx]
            break
        total_sum += length

    return len(sents), sents

def saveCSVFile(baseDir, media, article_dist):
    save_path = os.path.join(baseDir, media) + ".csv"

    article_dist.to_csv(save_path, mode='w', header=False)

def get_media_name(filepath):
    filename = filepath.split(os.sep)[-1]
    return filename.split(".")[0]

def is_small_text(lines):
    if not lines : return True
    return get_token_count(lines) < MIN_COUNT

def is_small_title(title):
    if title is '' : return True
    return len(title.split()) < TITLE_MIN_COUNT


def data_preprocessing(article_path, media_name):
    
    """Returns dists that added pre-processed data
    Args:
        article_path (str): origin articles file path
        media_name (str): origin article's media name
    Returns:
        dataframes : article's content summary dist, article's pre-processed summary dist, 
                    article's pre-processed title dist, article's pre-processed content dist
    """
    
    f = open(article_path, 'r', newline="\n", encoding="utf-8")

    processed_dist = pd.DataFrame(columns=['Title', 'Contents'])
    title_proc_dist = pd.DataFrame(columns=['Title', 'Contents'])
    summary_dist = pd.DataFrame(columns=['Title', 'Contents'])
    summary_proc_dist = pd.DataFrame(columns=['Title', 'Contents'])

    for [title, contents] in csv.reader(f):
        article = Article(title, media_name, contents.split("\t"))
            
        try:
            contents = list(article.readContent())

            # Preprocessing origin content and title 
            clean_conts = preprocessor.del_personal_info(contents, media_name)
            clean_conts = preprocessor.cleanLines(clean_conts)
            clean_title = preprocessor.cleanLine(article.title)

            # If input sequence length is longer than MAX_COUNT, split content
            split_idx, clean_conts = split_by_max_token(clean_conts)
            split_conts = contents[:split_idx]

            # Summarize article content
            conts_line = " ".join(split_conts)
            summary_lines = summarize(conts_line, ratio=0.2, split=True)

            # Preprocessing content summary
            summary_procs = preprocessor.del_personal_info(summary_lines, media_name)
            summary_procs = preprocessor.cleanLines(summary_procs)

            # If input sequence length is less than MIN_COUNT, drop article
            if is_small_text(clean_conts) or is_small_text(summary_procs) or is_small_title(clean_title): continue

            # Append article summary
            summary= {'Title' : article.title, 'Contents' : '\t'.join(summary_lines) }
            summary_dist = summary_dist.append(summary, ignore_index=True)
                    
            # Append article preprocessed summary
            summary_proc= {'Title' : article.title, 'Contents' : '\t'.join(summary_procs) }
            summary_proc_dist = summary_proc_dist.append(summary_proc, ignore_index=True)

            # Append article preprocessed title
            title_proc= {'Title' : article.title, 'Contents' : clean_title }
            title_proc_dist = title_proc_dist.append(title_proc, ignore_index=True)
                    
            # Append article preprocessed content
            proc = {'Title' : article.title, 'Contents' : '\t'.join(clean_conts)}
            processed_dist = processed_dist.append(proc, ignore_index=True)
                
            # print("Append : {title}\r\n{new}".format(title=article.title, new=clean_title))
                
        except Exception as err:
            print(err)
            # print("Drop Article : {title}".format(title=article.title))
            pass

    f.close()

    return summary_dist, summary_proc_dist, title_proc_dist, processed_dist

if __name__ == '__main__':

    print("Delete Remain Data")
    # Initialize preprocessed file
    del_folder(TITLE_PREPROCESSED_PATH)
    del_folder(PREPROCESSED_PATH)
    del_folder(SUMMARY_PATH)
    del_folder(SUMMARY_PREPROCESSED_PATH)
    del_folder(VAL_PREPROCESSED_PATH)
    del_folder(VAL_SUMMARY_PATH)
    del_folder(VAL_SUMMARY_PREPROCESSED_PATH)
    del_folder(VAL_TITLE_PREPROCESSED_PATH)

    mkdir_p(TITLE_PREPROCESSED_PATH)
    mkdir_p(PREPROCESSED_PATH)
    mkdir_p(SUMMARY_PATH)
    mkdir_p(SUMMARY_PREPROCESSED_PATH)
    mkdir_p(VAL_PREPROCESSED_PATH)
    mkdir_p(VAL_SUMMARY_PATH)
    mkdir_p(VAL_SUMMARY_PREPROCESSED_PATH)
    mkdir_p(VAL_TITLE_PREPROCESSED_PATH)

    print("Load Sword")
    # Load text preprocessor
    preprocessor = TextPreprocessor()
    preprocessor.loadSwords(SWORDS_PATH)

    # Pre-processing Origin-Data
    for idx, media_path in enumerate(iglob(os.path.join(ORIGIN_PATH, '**.csv'), recursive=False)):

        media_name = get_media_name(media_path)
        summary_dist, summary_proc_dist, title_proc_dist, processed_dist = data_preprocessing(media_path, media_name)

        # Save dist as csv file
        saveCSVFile(TITLE_PREPROCESSED_PATH, media_name, title_proc_dist)
        saveCSVFile(PREPROCESSED_PATH, media_name, processed_dist)
        saveCSVFile(SUMMARY_PREPROCESSED_PATH, media_name, summary_proc_dist)
        saveCSVFile(SUMMARY_PATH, media_name, summary_dist)

    # Pre-processing Valid-Data (for Evaluate trained model)
    for idx, media_path in enumerate(iglob(os.path.join(VALID_PATH, '**.csv'), recursive=False)):

        media_name = get_media_name(media_path)
        summary_dist, summary_proc_dist, title_proc_dist, processed_dist = data_preprocessing(media_path, media_name)

        # Save dist as csv file
        saveCSVFile(VAL_TITLE_PREPROCESSED_PATH, media_name, title_proc_dist)
        saveCSVFile(VAL_PREPROCESSED_PATH, media_name, processed_dist)
        saveCSVFile(VAL_SUMMARY_PREPROCESSED_PATH, media_name, summary_proc_dist)
        saveCSVFile(VAL_SUMMARY_PATH, media_name, summary_dist)
