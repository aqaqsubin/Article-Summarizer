import os
import re
import csv
from glob import iglob
from pathlib import Path
import numpy as np
import sentencepiece as spm
from module.dirHandler import mkdir_p, del_folder

BASE_DIR = "/data/ksb/"
DATA_BASE_DIR = os.path.join(BASE_DIR, "sample_articles")

ORIGIN_PATH = os.path.join(DATA_BASE_DIR,"Origin-Data")
PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Preprocessed-Data")
TITLE_PREPROCESSED_PATH = os.path.join(BASE_DIR,"Title-Preprocessed-Data")

PRETTY_PATH = os.path.join(DATA_BASE_DIR,"Pretty-Data")
VAL_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Valid-Preprocessed-Data")
VAL_SUMMARY_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Valid-Summary-Preprocessed-Data")
SUMMARY_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Summary-Preprocessed-Data")
SWORDS_PATH = os.path.join(DATA_BASE_DIR, "StopWordList.txt")
MODEL_DIR_PATH = "Word-Encoding-Model"

PAD_ID=0  
VACAB_SIZE = 80000 
SOS_ID=1
EOS_ID=2
UNK_ID=3
CHAR_COVERAGE = 1.0
MODEL_TYPE ='bpe' 

def get_cmd(filename, pad_id, sos_id, eos_id, unk_id, prefix, vocab_size, character_coverage, model_type):
    templates= '--input={} \
    --pad_id={} \
    --bos_id={} \
    --eos_id={} \
    --unk_id={} \
    --model_prefix={} \
    --vocab_size={} \
    --character_coverage={} \
    --model_type={}'
    
    cmd = templates.format(file_name,
                pad_id,
                sos_id,
                eos_id,
                unk_id,
                prefix,
                vocab_size,
                character_coverage,
                model_type)
    return cmd
    
def get_text(basepath):
    result = []

    for idx, proc_article_path in enumerate(iglob(os.path.join(basepath, '**.csv'), recursive=False)):
    
        f_proc= open(proc_article_path, 'r', newline="\n", encoding="utf-8")
        for [idx, title, contents] in csv.reader(f_proc):
            if contents is '': continue

            cont_list = contents.split("\t")
            result.append('\n'.join(cont_list))
        f_proc.close()

    return result

if __name__ == '__main__':

    mkdir_p(MODEL_DIR_PATH)

    file_name = os.path.join(MODEL_PATH, "SentencePiece.txt")

    headline_src_text = get_text(TITLE_PREPROCESSED_PATH)
    headline_tar_text = get_text(VAL_PREPROCESSED_PATH) # 원래 Generated Summary 
    src_text = get_text(PREPROCESSED_PATH)

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write('\n'.join(headline_src_text + headline_tar_text + src_text))

    model_num = len(list(iglob(os.path.join(MODEL_PATH, 'spm-input-*.vocab'), recursive=False)))
    prefix = os.path.join(MODEL_PATH, 'spm-input-{}'.format(model_num))

    src_cmd = get_cmd(file_name, PAD_ID, SOS_ID,
                EOS_ID, UNK_ID, prefix, VOCAB_SIZE, CHAR_COVERAGE, MODEL_TYPE)
    
    spm.SentencePieceTrainer.Train(src_cmd)