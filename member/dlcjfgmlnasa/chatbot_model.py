# chatbot_model.py

import tensorflow as tf
from tensorflow.contrib import rnn
from model.data_helper import PreDataProcessing


class Seq2Seq(object):
    def __init__(self,
                 encoder_size,
                 decoder_size,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 encoder_layer_size,
                 decoder_layer_size,
                 RNN_type='LSTM',
                 encoder_input_keep_prob=1.0,
                 encoder_output_keep_prob=1.0,
                 decoder_input_keep_prob=1.0,
                 decoder_output_keep_prob=1.0,
                 learning_rate=0.01,
                 hidden_size=128):

        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.encoder_layer_size = encoder_layer_size
        self.decoder_layer_size = decoder_layer_size
        self.encoder_input_keep_prob = encoder_input_keep_prob
        self.encoder_output_keep_prob = encoder_output_keep_prob
        self.decoder_input_keep_prob = decoder_input_keep_prob
        self.decoder_output_keep_prob = decoder_output_keep_prob
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size

        self.encoder_input = tf.placeholder(tf.float32, shape=(None, self.encoder_size, self.encoder_vocab_size))
        self.decoder_input = tf.placeholder(tf.float32, shape=(None, self.decoder_size, self.decoder_vocab_size))
        self.target_input = tf.placeholder(tf.int32, shape=(None, self.decoder_size))

        self.weight = tf.get_variable(shape=[self.hidden_size, self.decoder_vocab_size],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      dtype=tf.float32,
                                      name='weight')
        self.bias = tf.get_variable(shape=[self.decoder_vocab_size],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32,
                                    name='bias')

        self.logits = None
        self.cost = None
        self.train_op = None
        self.RNNCell = None
        self.outputs = None

        if RNN_type == 'LSTM':
            self.RNNCell = rnn.LSTMCell
        elif RNN_type == 'GRU':
            self.RNNCell = rnn.GRUCell
        else:
            raise Exception('not support {} RNN type'.format(RNN_type))

        self.build_model()

    def build_model(self):
        encoder_cell, decoder_cell = self.build_cells()

        with tf.variable_scope('encode'):
            outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_input, dtype=tf.float32)

        with tf.variable_scope('decode'):
            outputs, decoder_state = tf.nn.dynamic_rnn(decoder_cell, self.decoder_input,
                                                       initial_state=encoder_state, dtype=tf.float32)
        self.logits, self.cost, self.train_op = self.build_ops(outputs, self.target_input)
        self.outputs = tf.argmax(self.logits, 2)

    def build_cells(self):
        # encoder cell
        encoder_cell = self.RNNCell(num_units=self.hidden_size)
        encoder_cell = rnn.DropoutWrapper(encoder_cell,
                                          input_keep_prob=self.encoder_input_keep_prob,
                                          output_keep_prob=self.encoder_output_keep_prob)
        # encoder_cell = rnn.MultiRNNCell([encoder_cell for _ in range(self.encoder_layer_size)])

        # decoder cell
        decoder_cell = self.RNNCell(num_units=self.hidden_size)
        decoder_cell = rnn.DropoutWrapper(decoder_cell,
                                          input_keep_prob=self.decoder_input_keep_prob,
                                          output_keep_prob=self.decoder_output_keep_prob)
        # decoder_cell = rnn.MultiRNNCell([decoder_cell for _ in range(self.decoder_layer_size)])

        return encoder_cell, decoder_cell

    def build_ops(self, outputs, targets):
        time_steps = tf.shape(outputs)[1]
        outputs = tf.reshape(outputs, [-1, self.hidden_size])

        logits = tf.matmul(outputs, self.weight) + self.bias
        logits = tf.reshape(logits, [-1, time_steps, self.decoder_vocab_size])

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        return logits, cost, train_op

    def train(self, session, enc_input, dec_input, tar_input):
        return session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.encoder_input: enc_input,
                self.decoder_input: dec_input,
                self.target_input: tar_input
            }
        )


    def test(self, session, enc_input, dec_input, tar_input):
        pass

if __name__ == '__main__':
    hidden_layer = 128
    predataprocessing = PreDataProcessing()
    predataprocessing.load_file_dir('../data/train')
    predataprocessing.make_data_set()
    batchs = predataprocessing.iter_batch(epochs=100)
    encoder_vocab_size = predataprocessing.get_encoder_vocab_size()
    decoder_vocab_size = predataprocessing.get_decoder_vocab_size()
    encoder_size = predataprocessing.get_encoder_size()
    decoder_size = predataprocessing.get_decoder_size()

    s2s = Seq2Seq(
        encoder_size=encoder_size,
        decoder_size=decoder_size,
        encoder_vocab_size=encoder_vocab_size,
        decoder_vocab_size=decoder_vocab_size,
        encoder_layer_size=3,
        decoder_layer_size=3,
        RNN_type='GRU'
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch in batchs:
            enc_input, dec_input, tar_input = batch[0], batch[1], batch[2]
            cost, _ = s2s.train(sess, enc_input, dec_input, tar_input)
            print(cost)