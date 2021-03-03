
import os
import csv
import sys
import time
import argparse
import numpy as np
import pandas as pd
from glob import iglob
import tensorflow as tf
import sentencepiece as spm

sys.path.append("..")

from model.seq2seq import Encoder, Decoder

from module.dirHandler import mkdir_p, del_folder
from module.encoder import IntegerEncoder
from module.decoder import Decoder as IntegerDecoder
from module.parse import ParseBoolean
from sklearn.model_selection import train_test_split

BASE_DIR = os.getcwd()
DATA_BASE_DIR = os.path.join(BASE_DIR, 'articles')
SRC_BASE_DIR = os.path.join(BASE_DIR, 'src')

TITLE_PREPROCESSED_PATH= os.path.join(DATA_BASE_DIR,"Title-Preprocessed-Data")
PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Preprocessed-Data")

SUMMARY_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Summary-Preprocessed-Data")
PREDICT_PATH = os.path.join(DATA_BASE_DIR,"Predict-Data")
PREDICT_HEADLINE_PATH = os.path.join(DATA_BASE_DIR,"Predict-Headline-Data")
VAL_PREPROCESSED_PATH= os.path.join(DATA_BASE_DIR,"Valid-Preprocessed-Data")
VAL_SUMMARY_PREPROCESSED_PATH = os.path.join(DATA_BASE_DIR,"Valid-Summary-Preprocessed-Data")

WORD_ENCODING_DIR = os.path.join(SRC_BASE_DIR, 'Word-Encoding-Model')
MODEL_DIR = os.path.join(SRC_BASE_DIR, "trained-model")

parser = argparse.ArgumentParser(description="Description")
parser.add_argument('--headline', required=True, type=ParseBoolean, help="If True, Generating Headline else Generating Summary")
parser.add_argument('--n', required=True, type=int, help="Transformer + RC-Encoder (n)")

args = parser.parse_args()

sp = spm.SentencePieceProcessor()
model_num = len(list(iglob(os.path.join(WORD_ENCODING_DIR, 'spm-input-*.vocab'), recursive=False))) -1
with open(os.path.join(WORD_ENCODING_DIR, 'spm-input-{}.vocab'.format(model_num)), encoding='utf-8') as f:
    Vo = [doc.strip().split("\t") for doc in f]
sp.Load(os.path.join(WORD_ENCODING_DIR, 'spm-input-{}.model').format(model_num))

VOCAB_SIZE = len(Vo)
D_MODEL = 128

BATCH_SIZE = 64
BUFFER_SIZE = 5000

EPOCHS = 30
ENC_UNITS = 128
DEC_UNITS = ENC_UNITS * 2

START_TOKEN = [sp.bos_id()]
END_TOKEN = [sp.eos_id()]

EPOCHS = 30
BATCH_SIZE = 64

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp)

        dec_hidden = enc_hidden[0]


        dec_input = tf.expand_dims(START_TOKEN * BATCH_SIZE, 1)

        # 교사 강요(teacher forcing) - 다음 입력으로 타겟을 피딩(feeding)합니다.
        for t in range(1, targ.shape[1]):
            # enc_output를 디코더에 전달합니다.
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)

            # 교사 강요(teacher forcing)를 사용합니다.
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

if __name__ == '__main__':


    options = {
        'model-type' : 'Sentence-Piece',
        'inv_wv' : None,
        'corpus' : None,
        'spm' : sp
    }
    src_data_path = PREPROCESSED_PATH
    if args.headline:
        target_data_path = TITLE_PREPROCESSED_PATH
    else :
        target_data_path = SUMMARY_PREPROCESSED_PATH

    input_encoded_list = IntegerEncoder(options=options, filepaths=list(iglob(os.path.join(src_data_path, '**.csv'), recursive=False))).encoder()
    output_encoded_list = IntegerEncoder(options=options, filepaths=list(iglob(os.path.join(target_data_path, '**.csv'), recursive=False))).encoder()

    get_max_length = lambda x : np.max([len(line) for line in x])

    MAX_LEN = get_max_length(input_encoded_list) + 2
    SUMMARY_MAX_LEN = get_max_length(output_encoded_list) + 2

    input_encoded_list = list(map(lambda list_ : START_TOKEN + list_ + END_TOKEN, input_encoded_list))
    output_encoded_list = list(map(lambda list_ : START_TOKEN + list_ + END_TOKEN, output_encoded_list))
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

    dataset = tf.data.Dataset.from_tensor_slices((train_input_encoded_matrix, train_summary_encoded_matrix))
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    example_input_batch, example_target_batch = next(iter(dataset))
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    encoder = Encoder(VOCAB_SIZE, D_MODEL, ENC_UNITS, BATCH_SIZE, cell='lstm')
    sample_output, sample_hidden = encoder(example_input_batch)
    print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden[0].shape))
    print ('Encoder Cell state shape: (batch size, units) {}'.format(sample_hidden[1].shape))

    decoder = Decoder(VOCAB_SIZE, D_MODEL, DEC_UNITS, BATCH_SIZE, cell='lstm')

    hidden = sample_hidden[0]
    sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                        hidden, sample_output)

    print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

    mkdir_p(checkpoint_dirpath)
    checkpoint_prefix = os.path.join(MODEL_DIR, "Seq2Seq")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)

    
    BUFFER_SIZE = train_input_encoded_matrix.shape[0]
    steps_per_epoch = BUFFER_SIZE // BATCH_SIZE

    for epoch in range(EPOCHS): 
        start = time.time()

        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                        batch,
                                                        batch_loss.numpy()))
        # 에포크가 2번 실행될때마다 모델 저장 (체크포인트)
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))