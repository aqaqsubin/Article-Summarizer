import re
import os
import sys
import numpy as np
from pathlib import Path
from glob import iglob
import tensorflow as tf
import sentencepiece as spm
import csv
import pandas as pd
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.trans_rc_enc import transformer, CustomSchedule
from module.dirHandler import mkdir_p, del_folder
from module.encoder import IntegerEncoder
from module.decoder import Decoder
from module.parse import ParseBoolean
from sklearn.model_selection import train_test_split

# Get argument that determines the process and determines N
parser = argparse.ArgumentParser(description="Description")
parser.add_argument('--headline', required=True, type=ParseBoolean, help="If True, Generating Headline else Generating Summary")
parser.add_argument('--n', required=True, type=int, help="Transformer + RC-Encoder (n)")

args = parser.parse_args()

BASE_DIR = os.getcwd()
DATA_BASE_DIR = os.path.join(BASE_DIR, 'articles')
SRC_BASE_DIR = os.path.join(BASE_DIR, 'src')

VAL_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Valid-Preprocessed-Data")
VAL_SUMMARY_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Valid-Summary-Preprocessed-Data")
TITLE_PREPROCESSED_PATH= os.path.join(DATA_BASE_DIR,"Title-Preprocessed-Data")

PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Preprocessed-Data")
SUMMARY_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Summary-Preprocessed-Data")

TRANSFORMER_PREDICT_PATH = os.path.join(DATA_BASE_DIR,"Transformer-Predict-Data")

WORD_ENCODING_DIR = os.path.join(SRC_BASE_DIR, 'Word-Encoding-Model')
MODEL_DIR = os.path.join(SRC_BASE_DIR, 'trained-model')
TRAS_RC_ENC_MODEL_DIR = os.path.join(MODEL_DIR, 'Transformer_RC_Encoder_{}'.format(int(args.n)))


# Load Sentencepiece word encoding model
sp = spm.SentencePieceProcessor()
model_num = len(list(iglob(os.path.join(WORD_ENCODING_DIR, 'spm-input-*.vocab'), recursive=False))) -1
with open(os.path.join(WORD_ENCODING_DIR, 'spm-input-{}.vocab'.format(model_num)), encoding='utf-8') as f:
    Vo = [doc.strip().split("\t") for doc in f]

sp.Load(os.path.join(WORD_ENCODING_DIR, 'spm-input-{}.model').format(model_num))

D_MODEL = 128
VOCAB_SIZE = len(Vo)
LAYER_NUM = 6
RC_ENC_N = int(args.n)
NUM_HEADS = 8
DFF = 512

BATCH_SIZE = 64
BUFFER_SIZE = 5000

WARMUP_STEPS = 50
EPOCHS = 30
START_TOKEN = [sp.bos_id()]
END_TOKEN = [sp.eos_id()]

get_max_length = lambda x : np.max([len(line) for line in x])

MAX_LEN = 300 + 2
SUMMARY_MAX_LEN = 150 + 2

def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LEN-1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


if __name__ == '__main__':

    options = {
        'model-type' : 'Sentence-Piece',
        'inv_wv' : None,
        'corpus' : None,
        'spm' : sp
    }

    # src & target path depend on the process
    src_data_path = PREPROCESSED_PATH
    if args.headline:
        target_data_path = TITLE_PREPROCESSED_PATH
    else :
        target_data_path = SUMMARY_PREPROCESSED_PATH

    # Load src & target data for training model
    # src & target data integer encoding 
    input_encoded_list = IntegerEncoder(options=options, filepaths=list(iglob(os.path.join(src_data_path, '**.csv'), recursive=False))).encoder()
    output_encoded_list = IntegerEncoder(options=options, filepaths=list(iglob(os.path.join(target_data_path, '**.csv'), recursive=False))).encoder()

    MAX_LEN = get_max_length(input_encoded_list) + 2
    SUMMARY_MAX_LEN = get_max_length(output_encoded_list) + 2

    # add SOS & EOS Token (Start of Sentence, End of Sentence)
    input_encoded_list = list(map(lambda list_ : START_TOKEN + list_ + END_TOKEN, input_encoded_list))
    output_encoded_list = list(map(lambda list_ : START_TOKEN + list_ + END_TOKEN, output_encoded_list))

    # Divide into Train dataset & Validation dataset
    input_train, input_test, output_train, output_test = train_test_split(
        input_encoded_list, output_encoded_list, test_size=0.2, random_state=42)

    # Padding
    train_input_encoded_matrix = tf.keras.preprocessing.sequence.pad_sequences(
        input_train, maxlen=MAX_LEN, padding='post')
    train_summary_encoded_matrix = tf.keras.preprocessing.sequence.pad_sequences(
        output_train, maxlen=MAX_LEN, padding='post')
    test_input_encoded_matrix = tf.keras.preprocessing.sequence.pad_sequences(
        input_test, maxlen=MAX_LEN, padding='post')
    test_summary_encoded_matrix = tf.keras.preprocessing.sequence.pad_sequences(
        output_test, maxlen=MAX_LEN, padding='post')

    print('Train Contents Shape : {}'.format(train_input_encoded_matrix.shape))
    print('Train Summaries Shape : {}'.format(train_summary_encoded_matrix.shape))
    print('Test Contents Shape : {}'.format(test_input_encoded_matrix.shape))
    print('Test Summaries Shape : {}'.format(test_summary_encoded_matrix.shape))
    
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': train_input_encoded_matrix, # Encoder Input
            'dec_inputs': train_summary_encoded_matrix[:, :-1] # Decoder Input
        },
        {
            # Decoder Output, Remove <SOS>
            'outputs': train_summary_encoded_matrix[:, 1:]  
        },
    ))
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': test_input_encoded_matrix, # Encoder Input
            'dec_inputs': test_summary_encoded_matrix[:, :-1] # Decoder Input
        },
        {
            # Decoder Output, Remove <SOS>
            'outputs': test_summary_encoded_matrix[:, 1:]  
        },
    ))
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.shuffle(BUFFER_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Declaring optimizer
    lrate_scheduler = CustomSchedule(d_model=D_MODEL)
    beta_1 = 0.9  
    beta_2 = 0.98
    epsilon = 10 ** -9

    optimizer = tf.keras.optimizers.Adam(lrate_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Initialize Transformer
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=LAYER_NUM,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout = 0.3,
        rc_enc_N = RC_ENC_N)

    
    # Initialize model train checkpoint
    mkdir_p(TRAS_RC_ENC_MODEL_DIR)
    checkpoint_filepath = os.path.join(TRAS_RC_ENC_MODEL_DIR, "checkpoint.ckpt")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='max',
        save_best_only=True)
    
    # Training Model
    model.compile(optimizer=optimizer, loss=loss_function)
    model.summary()

    model.fit(dataset, batch_size=BATCH_SIZE, epochs=30, verbose=2, validation_data=val_dataset, shuffle=True, callbacks=[model_checkpoint_callback])
