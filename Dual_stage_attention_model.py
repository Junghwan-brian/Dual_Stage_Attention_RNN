import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class Encoderlstm(Layer):
    def __init__(self, m):
        """
        m : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Encoderlstm, self).__init__(name="encoder_lstm")
        self.lstm = LSTM(m, return_state=True)
        self.initial_state = None

    def call(self, x, training=False):
        """
        x : t 번째 input data (shape = batch,1,n)
        """
        h_s, _, c_s = self.lstm(x, initial_state=self.initial_state)
        self.initial_state = [h_s, c_s]
        return h_s, c_s

    def reset_state(self, h0, c0):
        self.initial_state = [h0, c0]


class InputAttention(Layer):
    def __init__(self, T):
        super(InputAttention, self).__init__(name="input_attention")
        self.w1 = Dense(T)
        self.w2 = Dense(T)
        self.v = Dense(1)

    def call(self, h_s, c_s, x):
        """
        h_s : hidden_state (shape = batch,m)
        c_s : cell_state (shape = batch,m)
        x : time series encoder inputs (shape = batch,T,n)
        """
        query = tf.concat([h_s, c_s], axis=-1)  # batch, m*2
        query = RepeatVector(x.shape[2])(query)  # batch, n, m*2
        x_perm = Permute((2, 1))(x)  # batch, n, T
        score = tf.nn.tanh(self.w1(x_perm) + self.w2(query))  # batch, n, T
        score = self.v(score)  # batch, n, 1
        score = Permute((2, 1))(score)  # batch,1,n
        attention_weights = tf.nn.softmax(score)  # t 번째 time step 일 때 각 feature 별 중요도
        return attention_weights


class Encoder(Layer):
    def __init__(self, T, m):
        super(Encoder, self).__init__(name="encoder")
        self.T = T
        self.input_att = InputAttention(T)
        self.lstm = Encoderlstm(m)
        self.initial_state = None
        self.alpha_t = None

    def call(self, data, h0, c0, n=39, training=False):
        """
        data : encoder data (shape = batch, T, n)
        n : data feature num
        """
        self.lstm.reset_state(h0=h0, c0=c0)
        alpha_seq = tf.TensorArray(tf.float32, self.T)
        for t in range(self.T):
            x = Lambda(lambda x: data[:, t, :])(data)
            x = x[:, tf.newaxis, :]  # (batch,1,n)

            h_s, c_s = self.lstm(x)

            self.alpha_t = self.input_att(h_s, c_s, data)  # batch,1,n

            alpha_seq = alpha_seq.write(t, self.alpha_t)
        alpha_seq = tf.reshape(alpha_seq.stack(), (-1, self.T, n))  # batch, T, n
        output = tf.multiply(data, alpha_seq)  # batch, T, n

        return output


class Decoderlstm(Layer):
    def __init__(self, p):
        """
        p : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Decoderlstm, self).__init__(name="decoder_lstm")
        self.lstm = LSTM(p, return_state=True)
        self.initial_state = None

    def call(self, x, training=False):
        """
        x : t 번째 input data (shape = batch,1,n)
        """
        h_s, _, c_s = self.lstm(x, initial_state=self.initial_state)
        self.initial_state = [h_s, c_s]
        return h_s, c_s

    def reset_state(self, h0, c0):
        self.initial_state = [h0, c0]


class TemporalAttention(Layer):
    def __init__(self, m):
        super(TemporalAttention, self).__init__(name="temporal_attention")
        self.w1 = Dense(m)
        self.w2 = Dense(m)
        self.v = Dense(1)

    def call(self, h_s, c_s, enc_h):
        """
        h_s : hidden_state (shape = batch,p)
        c_s : cell_state (shape = batch,p)
        enc_h : time series encoder inputs (shape = batch,T,m)
        """
        query = tf.concat([h_s, c_s], axis=-1)  # batch, p*2
        query = RepeatVector(enc_h.shape[1])(query)
        score = tf.nn.tanh(self.w1(enc_h) + self.w2(query))  # batch, T, m
        score = self.v(score)  # batch, T, 1
        attention_weights = tf.nn.softmax(
            score, axis=1
        )  # encoder hidden state h(i) 의 중요성 (0<=i<=T)
        return attention_weights


class Decoder(Layer):
    def __init__(self, T, p, m):
        super(Decoder, self).__init__(name="decoder")
        self.T = T
        self.temp_att = TemporalAttention(m)
        self.dense = Dense(1)
        self.lstm = Decoderlstm(p)
        self.enc_lstm_dim = m
        self.dec_lstm_dim = p
        self.context_v = None
        self.dec_h_s = None
        self.beta_t = None

    def call(self, data, enc_h, h0=None, c0=None, training=False):
        """
        data : decoder data (shape = batch, T-1, 1)
        enc_h : encoder hidden state (shape = batch, T, m)
        """
        h_s = None
        self.lstm.reset_state(h0=h0, c0=c0)
        self.context_v = tf.zeros((enc_h.shape[0], 1, self.enc_lstm_dim))  # batch,1,m
        self.dec_h_s = tf.zeros((enc_h.shape[0], self.dec_lstm_dim))  # batch, p
        for t in range(self.T - 1):  # 0~T-1
            x = Lambda(lambda x: data[:, t, :])(data)
            x = x[:, tf.newaxis, :]  #  (batch,1,1)
            x = tf.concat([x, self.context_v], axis=-1)  # batch, 1, m+1
            x = self.dense(x)  # batch,1,1

            h_s, c_s = self.lstm(x)  # batch,p

            self.beta_t = self.temp_att(h_s, c_s, enc_h)  # batch, T, 1

            self.context_v = tf.matmul(
                self.beta_t, enc_h, transpose_a=True
            )  # batch,1,m
        return tf.concat(
            [h_s[:, tf.newaxis, :], self.context_v], axis=-1
        )  # batch,1,m+p


class DARNN(Model):
    def __init__(self, T, m, p):
        super(DARNN, self).__init__(name="DARNN")
        """
        T : 주기 (time series length)
        m : encoder lstm feature length
        p : decoder lstm feature length
        h0 : lstm initial hidden state
        c0 : lstm initial cell state
        """
        self.m = m
        self.encoder = Encoder(T=T, m=m)
        self.decoder = Decoder(T=T, p=p, m=m)
        self.lstm = LSTM(m, return_sequences=True)
        self.dense1 = Dense(p)
        self.dense2 = Dense(1)

    def call(self, inputs, training=False, mask=None):
        """
        inputs : [enc , dec]
        enc_data : batch,T,n
        dec_data : batch,T-1,1
        """
        enc_data, dec_data = inputs
        batch = enc_data.shape[0]
        h0 = tf.zeros((batch, self.m))
        c0 = tf.zeros((batch, self.m))
        enc_output = self.encoder(
            enc_data, n=39, h0=h0, c0=c0, training=training
        )  # batch, T, n
        enc_h = self.lstm(enc_output)  # batch, T, m
        dec_output = self.decoder(
            dec_data, enc_h, h0=h0, c0=c0, training=training
        )  # batch,1,m+p
        output = self.dense2(self.dense1(dec_output))
        output = tf.squeeze(output)
        return output