import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Embedding, Dense, Dropout
import tensorflow_addons as tfa


class CWSModel(Model):
    def __init__(self, num_hidden, vocab_size, num_label, embed_table=None, embed_feature=300):
        super(CWSModel, self).__init__()

        self.num_hidden = num_hidden
        self.vocab_size = vocab_size
        self.num_label = num_label
        self.transition_params = None
        if embed_table == None:
            self.embed_table = Embedding(vocab_size, embed_feature)
        else:
            self.embed_table = Embedding(vocab_size, embed_feature, embeddings_initializer=embed_table)

        self.biLSTM = Bidirectional(LSTM(num_hidden, return_sequences=True))
        self.dense = Dense(num_label)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(5,5)), trainable=False)   # ckpt保留用
        self.dropout = Dropout(0.5)
    
    def call(self, input_sequences, label_sequences=None, training=False):
        # LSTM 的logits: [batchsz, seq_len, vocab_size]
        #        seq_lens: [batchsz,] batch中每个sequence 没有padding前的原始长度
        seq_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(input_sequences, 0), dtype=tf.int32), axis=-1)
        # mask = self.embed_table.compute_mask(input_sequences)
        # seq_lens = tf.math.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=-1)
        
        inputs = self.embed_table(input_sequences)
        inputs = self.dropout(inputs, training)
        logits = self.dense(self.biLSTM(inputs))

        if label_sequences is not None:
            # 需要确定dtype=int32, 默认dtype=int64会导致 crf计算出错
            label_sequences = tf.convert_to_tensor(label_sequences, dtype=tf.int32)          
            log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(logits, label_sequences, seq_lens)
            self.transition_params = tf.Variable(self.transition_params, trainable=False)       # ckpt不追溯 tensor, 而追溯variable
            return logits, seq_lens, log_likelihood
        else:
            return logits, seq_lens

        



