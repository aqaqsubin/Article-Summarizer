import numpy as np
import tensorflow as tf 
from tensorflow.keras.layers import LSTM, GRU, Activation, Conv1D, BatchNormalization,Dense,Bidirectional
from tensorflow.keras.models import Sequential


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, N, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(N, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, N, d_model):
        angle_rads = self.get_angles(   
            position=tf.range(N, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        sines = tf.math.sin(angle_rads[:, 0::2])

        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x) 
    return tf.maximum(look_ahead_mask, padding_mask)


def scaled_dot_product_attention(query, key, value, mask):

    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs): 
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)

        return outputs

def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
          'mask': padding_mask # 패딩 마스크 사용
      })

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
                                dropout=dropout, name="encoder_layer_{}".format(i),
                               )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


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


def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")

    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs, 'key': inputs, 'value': inputs, 
          'mask': look_ahead_mask 
      })

    attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 
          'mask': padding_mask 
      })

    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


def decoder_last_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
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
    print(gated.shape)
    print(encoder_decoder_attention.shape)
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

    for i in range(num_layers-1):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
                                dropout=dropout, name='decoder_layer_{}'.format(i),
                                )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
    
    outputs = decoder_last_layer(dff=dff, d_model=d_model, num_heads=num_heads,
                                dropout=dropout, name='decoder_last_layer',
                                )(inputs=[outputs, enc_outputs, rc_enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, rc_enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def transformer(vocab_size, num_layers, dff,
                d_model, num_heads, dropout,
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
        d_model=d_model, num_heads=num_heads, dropout=dropout,
        )(inputs=[dec_inputs, enc_outputs, rc_enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)