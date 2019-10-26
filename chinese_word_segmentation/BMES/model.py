# encoding=utf-8
import tensorflow as tf
from hparams import Hparams as hp


class BiLSTM_CRF():
    def __init__(self):
        self.sent_input = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_len])
        self.label = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_len])
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])
        self.name = "BiLSTM_CRF"

    def embedding(self, zero_pad=True):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            lookup_table = tf.get_variable("lookup_table",
                                           shape=[hp.vocab_size, hp.num_units],
                                           dtype=tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer())
            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, hp.num_units]),
                                          lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, self.sent_input)
            # outputs: [batch_size, max_len, num_units]
            return outputs

    def biLSTM(self, inputs):
        with tf.variable_scope("biLSTM", reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(hp.num_units)
            # cell_fw = tf.contrib.nn.DropoutWrapper()
            # 增加的dropout查看效果
            cell_bw = tf.contrib.rnn.BasicLSTMCell(hp.num_units)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                                    sequence_length=self.sequence_length,
                                                                    dtype=tf.float32)
            outputs = tf.concat([outputs[0], outputs[1]], axis=-1)
            # outputs: [batch_size, max_len, 2*num_units]
            return outputs, final_states

    def ff_crf(self, inputs):
        # fully connected networks
        with tf.variable_scope("ff", reuse=tf.AUTO_REUSE):
            W = tf.get_variable(
                name='W',
                shape=[2*hp.num_units, len(hp.tag2label)],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32
            )
            b = tf.get_variable(
                name='b',
                shape=[len(hp.tag2label)],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32
            )
            # s = tf.shape(inputs)
            outputs = tf.reshape(inputs, [-1, 2*hp.num_units])
            # outputs: [batch_size*max_len, 2*num_units]
            pred = tf.matmul(outputs, W) + b
            # pred: [batch_size*max_len, len(tag2label)]
            logits = tf.reshape(pred, [-1, hp.max_len, len(hp.tag2label)])
            # logits: [batch_size, max_len, len(tag2label)]
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.label,
                sequence_lengths=self.sequence_length
            )
            loss = -tf.reduce_mean(log_likelihood)
            return logits, loss, transition_params

    def train(self):
        embed = self.embedding()
        # embed: [batch_size, max_len, num_units]
        biLSTM_outputs, _ = self.biLSTM(embed)
        # biLSTM_outputs: [batch_size, max_len, 2*num_units]
        _, loss, _ = self.ff_crf(biLSTM_outputs)
        # logits: [batch_size, max_len, len(tag2label)]

        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return loss, train_op, global_step

    def eval(self):
        embed = self.embedding()
        biLSTM_outputs, _ = self.biLSTM(embed)
        logits, _ , transition_params= self.ff_crf(biLSTM_outputs)
        # label_pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        # label_pred: [batch_size, max_len]
        return logits, transition_params



