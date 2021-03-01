import tensorflow as tf
from tensorflow.keras.layers import Input
import numpy as np
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # 쿼리 은닉 상태(query hidden state)는 (batch_size, hidden size)쌍으로 이루어져 있습니다.
        # query_with_time_axis은 (batch_size, 1, hidden size)쌍으로 이루어져 있습니다.
        # values는 (batch_size, max_len, hidden size)쌍으로 이루어져 있습니다.
        # 스코어(score)계산을 위해 덧셈을 수행하고자 시간 축을 확장하여 아래의 과정을 수행합니다.
        query_with_time_axis = tf.expand_dims(query, 1)

        # score는 (batch_size, max_length, 1)쌍으로 이루어져 있습니다.
        # score를 self.V에 적용하기 때문에 마지막 축에 1을 얻습니다.
        # self.V에 적용하기 전에 텐서는 (batch_size, max_length, units)쌍으로 이루어져 있습니다.
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis)
                                  + self.W2(values)))

        # attention_weights는 (batch_size, max_length, 1)쌍으로 이루어져 있습니다. 
        attention_weights = tf.nn.softmax(score, axis=1)

        # 덧셈이후 컨텍스트 벡터(context_vector)는 (batch_size, hidden_size)쌍으로 이루어져 있습니다.        
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, cell='gru'):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.cell = cell
        if self.cell is 'gru':
            self.rnn =tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform'))
        else:
            self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform'))

    def call(self, x):
        x = self.embedding(x)
        if self.cell is 'gru':
            output, forward_h, backward_h = self.rnn(x)
            state = Concatenate()([forward_h, backward_h])
        else:
            output, forward_h, forward_c, backward_h, backward_c = self.rnn(x)
            state_h = Concatenate()([forward_h, backward_h])
            state_c = Concatenate()([forward_c, backward_c])
            
            state = [state_h, state_c]

        return output, state

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units))]


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, cell='gru'):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        self.cell = cell
        if self.cell is 'gru':
            self.rnn =tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        else:
            self.rnn = tf.keras.layers.LSTM(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
            
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, state_h, enc_output):
        # enc_output는 (batch_size, max_length, hidden_size)쌍으로 이루어져 있습니다.
        context_vector, attention_weights = self.attention(state_h, enc_output)


        # 임베딩층을 통과한 후 x는 (batch_size, 1, embedding_dim)쌍으로 이루어져 있습니다.
        x = self.embedding(x)

        # 컨텍스트 벡터와 임베딩 결과를 결합한 이후 x의 형태는 (batch_size, 1, embedding_dim + hidden_size)쌍으로 이루어져 있습니다.
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 위에서 결합된 벡터를 GRU에 전달합니다.
        if self.cell is 'gru':
            output, state = self.rnn(x, inital_)
        else :
            output, state, _ = self.rnn(x)

        # print("decoder output : {}".format(output.shape))

        # output은 (batch_size * 1, hidden_size)쌍으로 이루어져 있습니다.
        output = tf.reshape(output, (-1, output.shape[2]))
        # print("decoder reshape output : {}".format(output.shape))

        # output은 (batch_size, vocab)쌍으로 이루어져 있습니다.
        x = self.fc(output)
        # print("decoder final output : {}".format(x.shape))

        return x, state, attention_weights