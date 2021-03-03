import os
import re
import sys
import csv
import argparse
import pandas as pd
import numpy as np
from glob import iglob
import tensorflow as tf
import sentencepiece as spm
from rouge import Rouge 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.seq2seq import Encoder, Decoder
from module.dirHandler import mkdir_p, del_folder
from module.encoder import IntegerEncoder
from module.decoder import Decoder as IntegerDecoder
from module.parse import ParseBoolean

BASE_DIR = os.getcwd()
SRC_BASE_DIR = os.path.join(BASE_DIR, 'src')
DATA_BASE_DIR = os.path.join(BASE_DIR, 'articles')

PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Preprocessed-Data")
SUMMARY_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Summary-Preprocessed-Data")

SUMMARY_PREDICT_PATH = os.path.join(DATA_BASE_DIR,"Summary-Predict-Data")
TITLE_PREDICT_PATH = os.path.join(DATA_BASE_DIR,"Title-Predict-Data")

VAL_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Valid-Preprocessed-Data")
VAL_SUMMARY_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Valid-Summary-Preprocessed-Data")
VAL_TITLE_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Valid-Title-Preprocessed-Data")

SEQ2SEQ_PREDICT_PATH = os.path.join(SUMMARY_PREDICT_PATH,"Seq2Seq-Predict-Data")

WORD_ENCODING_DIR = os.path.join(SRC_BASE_DIR, 'Word-Encoding-Model')
MODEL_DIR = os.path.join(SRC_BASE_DIR, 'trained-model')
S2S_MODEL_DIR = os.path.join(MODEL_DIR, "seq2seq")

parser = argparse.ArgumentParser(description="Description")
parser.add_argument('--headline', required=True, type=ParseBoolean, help="If True, Generating Headline else Generating Summary")

args = parser.parse_args()

sp = spm.SentencePieceProcessor()
model_num = len(list(iglob(os.path.join(WORD_ENCODING_DIR, 'spm-input-*.vocab'), recursive=False))) -1
with open(os.path.join(WORD_ENCODING_DIR, 'spm-input-{}.vocab'.format(model_num)), encoding='utf-8') as f:
    Vo = [doc.strip().split("\t") for doc in f]
sp.Load(os.path.join(WORD_ENCODING_DIR, 'spm-input-{}.model').format(model_num))

VOCAB_SIZE = len(Vo)
D_MODEL = 128
SUMMARY_MAX_LEN = 150 + 2
START_TOKEN = [sp.bos_id()]
END_TOKEN = [sp.eos_id()]

LAYER_NUM = 6
NUM_HEADS = 8
DFF = 512

BATCH_SIZE = 64
BUFFER_SIZE = 5000

WARMUP_STEPS = 50
EPOCHS = 30

def evaluate(content):
    content = tf.expand_dims(content, axis=0)
    inputs = tf.convert_to_tensor(content)

    enc_out, enc_stats = encoder(inputs)
    
    enc_hidden_stat = enc_stats[0]
    enc_cell_stat = enc_stats[1]

    dec_hidden = enc_hidden_stat
    dec_input = tf.expand_dims(START_TOKEN, 0)

    idx_list = []
    for t in range(SUMMARY_MAX_LEN):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)

        predicted_id = int(tf.argmax(predictions[0]).numpy())
        idx_list.append(int(predicted_id))

        if [predicted_id] == END_TOKEN:
            return sp.decode_ids(idx_list), content 

        dec_input = tf.expand_dims([predicted_id], 0)

    return sp.decode_ids(idx_list) , content 

def summary(sentence):
    result, sentence = evaluate(sentence)
    return result

def saveCSVFile(baseDir, media, article_dist):
    save_path = os.path.join(baseDir, media) + ".csv"

    article_dist.to_csv(save_path, mode='w', header=False)

def get_media_name(filepath):
    filename = filepath.split(os.sep)[-1]
    return filename.split(".")[0]

def get_rouge_score(gen_summary, target_summary):
    
    try:
        rouge_scores = rouge.get_scores(predict_summary, target_summary)[0]
        rouge_L = rouge_scores['rouge-l']
        rouge_L_f = rouge_L['f']
        rouge_L_r = rouge_L['r']
        rouge_L_p = rouge_L['p']

        print('ROUGE-L F1: {}'.format(rouge_L_f * 100))
        print('ROUGE-L Recall: {}'.format(rouge_L_r * 100))
        print('ROUGE-L Precision: {}\n'.format(rouge_L_p * 100))

        rouge_1 = rouge_scores['rouge-1']
        rouge_1_f = rouge_1['f']
        rouge_1_r = rouge_1['r']
        rouge_1_p = rouge_1['p']

        print('ROUGE-1 F1: {}'.format(rouge_1_f * 100))
        print('ROUGE-1 Recall: {}'.format(rouge_1_r * 100))
        print('ROUGE-1 Precision: {}\n'.format(rouge_1_p * 100))

        rouge_2 = rouge_scores['rouge-2']
        rouge_2_f = rouge_2['f']
        rouge_2_r = rouge_2['r']
        rouge_2_p = rouge_2['p']

        print('ROUGE-2 F1: {}'.format(rouge_2_f * 100))
        print('ROUGE-2 Recall: {}'.format(rouge_2_r * 100))
        print('ROUGE-2 Precision: {}\n'.format(rouge_2_p * 100))


    except IndexError as e:
        print(e)
        rouge_L_f = rouge_L_r = rouge_L_p = 0.0
        rouge_1_f = rouge_1_r = rouge_1_p = 0.0
        rouge_2_f = rouge_2_r = rouge_2_p = 0.0

    
    score = {'ROUGE-L F1': rouge_L_f, 'ROUGE-L Recall': rouge_L_r, 'ROUGE-L Precision': rouge_L_p,
            'ROUGE-1 F1': rouge_1_f, 'ROUGE-1 Recall': rouge_1_r, 'ROUGE-1 Precision': rouge_1_p,
            'ROUGE-2 F1': rouge_2_f, 'ROUGE-2 Recall': rouge_2_r, 'ROUGE-2 Precision': rouge_2_p}
        
    return score


if __name__ == '__main__':

    options = {
        'model-type' : 'Sentence-Piece',
        'inv_wv' : None,
        'corpus' : None,
        'spm' : sp
    }

    encoder = Encoder(VOCAB_SIZE, D_MODEL, ENC_UNITS, BATCH_SIZE, cell='lstm')
    decoder = Decoder(VOCAB_SIZE, D_MODEL, DEC_UNITS, BATCH_SIZE, cell='lstm')

    ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, S2S_MODEL_DIR, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)

    del_folder(SEQ2SEQ_PREDICT_PATH)
    mkdir_p(SEQ2SEQ_PREDICT_PATH)
    
    rouge = Rouge()
    
    if args.headline:
        src_data_path = VAL_SUMMARY_PREPROCESSED_PATH
        target_data_path = VAL_TITLE_PREPROCESSED_PATH
    else :
        src_data_path = VAL_PREPROCESSED_PATH
        target_data_path = VAL_SUMMARY_PREPROCESSED_PATH

    for _, val_proc_path in enumerate(iglob(os.path.join(src_data_path, '**.csv'), recursive=False)):

        media_name = get_media_name(val_proc_path)
        val_summary_path = os.path.join(target_data_path, media_name +".csv")
        print(media_name, val_proc_path)

        f_src = open(val_proc_path, 'r', newline="\n", encoding="utf-8")
        f_tar = open(val_summary_path, 'r', newline="\n", encoding="utf-8")

        validation_generated_dist = pd.DataFrame(columns=['Origin Contents', 'Target Summary', 'Generated Summary', 
                                                'ROUGE-L F1', 'ROUGE-L Recall', 'ROUGE-L Precision',
                                                'ROUGE-1 F1', 'ROUGE-1 Recall', 'ROUGE-1 Precision',
                                                'ROUGE-2 F1', 'ROUGE-2 Recall', 'ROUGE-2 Precision'])

        for [_, title, contents], [_, _, target] in zip(csv.reader(f_src), csv.reader(f_tar)):
            content = contents.split("\t")
            target_summary = target.split("\t")

            encoder = IntergerEncoder(options=option, filepaths=None)
            input_sent = ' '.join(content)
            input_enc_sent = START_TOKEN + encoder.line_encoder(content_line) + END_TOKEN

            predict_summary = summary(input_enc_sent)
            target_summary = ' '.join(target_summary)

            print('Input: {}'.format(input_sent))
            print('Target: {}'.format(target_summary))
            print('Output: {}'.format(predict_summary))

            rouge_scores = get_rouge_score(predict_summary, target_summary)
            
            summary = {'Origin Contents' : input_sent, 'Generated Summary' : predict_summary, 'Target Summary' : target_summary,
                   'ROUGE-L F1': rouge_scores['ROUGE-L F1'], 'ROUGE-L Recall': rouge_scores['ROUGE-L Recall'], 'ROUGE-L Precision': rouge_scores['ROUGE-L Precision'],
                   'ROUGE-1 F1': rouge_scores['ROUGE-1 F1'], 'ROUGE-1 Recall': rouge_scores['ROUGE-1 Recall'], 'ROUGE-1 Precision': rouge_scores['ROUGE-1 Precision'],
                   'ROUGE-2 F1': rouge_scores['ROUGE-2 F1'], 'ROUGE-2 Recall': rouge_scores['ROUGE-2 Recall'], 'ROUGE-2 Precision': rouge_scores['ROUGE-2 Precision']}

            validation_generated_dist = validation_generated_dist.append(summary, ignore_index=True)
            print("Current ROUGE-L mean : {}".format(np.mean(validation_generated_dist['ROUGE-L F1'])))

        print("ROUGE-1 mean : {}".format(np.mean(validation_generated_dist['ROUGE-1 F1'])))
        print("ROUGE-2 mean : {}".format(np.mean(validation_generated_dist['ROUGE-2 F1'])))
        print("ROUGE-L mean : {}".format(np.mean(validation_generated_dist['ROUGE-L F1'])))
        saveCSVFile(SEQ2SEQ_PREDICT_PATH, media_name, validation_generated_dist)