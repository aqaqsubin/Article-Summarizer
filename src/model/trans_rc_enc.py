import numpy as np
import tensorflow as tf
from model.transformer import PositionalEncoding, MultiHeadAttention, CustomSchedule
from model.transformer import create_padding_mask, create_look_ahead_mask
from model.transformer import encoder_layer, encoder, decoder_layer
from tensorflow.keras.layers import LSTM, GRU, Activation, Conv1D, BatchNormalization,Dense,Bidirectional
from tensorflow.keras.models import Sequential


def rc_encoder(vocab_size, d_model, hidden_size, encoder_input, global_layers=1, cell='gru', dropout=0.1):

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(encoder_input)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    inputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    sw1 = Sequential()
    sw1.add(Conv1D(filters=hidden_size, kernel_size =1, padding='same'))
    sw1.add(BatchNormalization())
    sw1.add(Activation('relu'))

    sw3 = Sequential()
    sw3.add(Conv1D(filters=hidden_size, kernel_size =1, padding='same'))
    sw3.add(Activation('relu'))
    sw3.add(BatchNormalization())
    sw3.add(Conv1D(filters=hidden_size, kernel_size =3, padding='same'))
    sw3.add(Activation('relu'))
    sw3.add(BatchNormalization())

    sw33 = Sequential()
    sw33.add(Conv1D(filters=hidden_size, kernel_size =1, padding='same'))
    sw33.add(Activation('relu'))
    sw33.add(BatchNormalization())
    sw33.add(Conv1D(filters=hidden_size, kernel_size =3, padding='same'))
    sw33.add(Activation('relu'))
    sw33.add(BatchNormalization())
    sw33.add(Conv1D(filters=hidden_size, kernel_size =3, padding='same'))
    sw33.add(Activation('relu'))
    sw33.add(BatchNormalization())

    filter_linear = Sequential()
    filter_linear.add(Dense(hidden_size, activation='sigmoid'))

    rnn = Sequential()
    if cell == 'gru':
        for layer_idx in range(global_layers):
            rnn.add(Bidirectional(GRU(units=hidden_size, dropout=dropout, return_sequences=True)))
                
    else:
        for layer_idx in range(global_layers):
            rnn.add(Bidirectional(LSTM(units=hidden_size, dropout=dropout, return_sequences=True)))
                
    rnn.add(Dense(units=hidden_size))
                                     
    outputs = rnn(inputs) # (Batch_size, Length, Hidden_size)
        
    conv1 = sw1(outputs) 
    conv3 = sw3(outputs)
    conv33 = sw33(outputs)
                             
    conv = tf.concat((conv1, conv3, conv33), -1) # (Batch_size, Length, 3 * Hidden_size)
    conv = filter_linear(conv) # (Batch_size, Length, Hidden_size)

    outputs = outputs * conv
    return outputs   


def rc_decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    decoder_input = tf.keras.Input(shape=(None, d_model), name="inputs")
    encoder_output = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    rc_encoder_output = tf.keras.Input(shape=(None, d_model), name="rc_encoder_outputs")

    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")

    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    query = key = value = decoder_input

    self_attention = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
          'query': query, 'key': key, 'value': value,
          'mask': look_ahead_mask
        })

    sublayer_output_1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(self_attention + decoder_input)
    
    key_from_encoder = value_from_encoder = encoder_output
    query_from_decoder = sublayer_output_1

    encoder_decoder_attention = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': query_from_decoder, 'key': key_from_encoder, 'value': value_from_encoder,
          'mask': padding_mask # 패딩 마스크
      })

    encoder_decoder_attention = tf.keras.layers.Dropout(rate=dropout)(encoder_decoder_attention)
    encoder_decoder_attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(encoder_decoder_attention + sublayer_output_1)

    key_from_rc_encoder = value_from_rc_encoder = rc_encoder_output

    rc_encoder_decoder_attention = MultiHeadAttention(
      d_model, num_heads, name="rc_attention_2")(inputs={
          'query': query_from_decoder, 'key': key_from_rc_encoder, 'value': value_from_rc_encoder,
          'mask': padding_mask # 패딩 마스크
      })

    rc_encoder_decoder_attention = tf.keras.layers.Dropout(rate=dropout)(rc_encoder_decoder_attention)
    rc_encoder_decoder_attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(rc_encoder_decoder_attention + sublayer_output_1)
    
    concat_attention = tf.concat([encoder_decoder_attention, rc_encoder_decoder_attention], -1)
    gated = tf.keras.layers.Dense(d_model, activation='sigmoid')(concat_attention)

    x = gated * encoder_decoder_attention
    y= (1 - gated)*rc_encoder_decoder_attention

    sublayer_output_2 = x + y

    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(sublayer_output_2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + sublayer_output_2)

    return tf.keras.Model(
      inputs=[decoder_input, encoder_output, rc_encoder_output, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


def decoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            rc_enc_N=0,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    rc_enc_outputs = tf.keras.Input(shape=(None, d_model), name='rc_encoder_outputs')

    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers - rc_enc_N):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
                                dropout=dropout, name='decoder_layer_{}'.format(i),
                                )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
    
    for i in range(rc_enc_N):
        outputs = rc_decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
                                dropout=dropout, name='decoder_{}_layer'.format(i),
                                )(inputs=[outputs, enc_outputs, rc_enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, rc_enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def transformer(vocab_size, num_layers, dff,
                d_model, num_heads, dropout, rc_enc_N= 0,
                name="transformer"):

    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
        d_model=d_model, num_heads=num_heads, dropout=dropout,
        )(inputs=[inputs, enc_padding_mask]) # 인코더의 입력은 입력 문장과 패딩 마스크

    rc_enc_outputs = rc_encoder(vocab_size=vocab_size, d_model=d_model, 
                                hidden_size=d_model,encoder_input=inputs, global_layers=1, cell='gru', dropout=dropout)

    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
        d_model=d_model, num_heads=num_heads, dropout=dropout, rc_enc_N=rc_enc_N,
        )(inputs=[dec_inputs, enc_outputs, rc_enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
