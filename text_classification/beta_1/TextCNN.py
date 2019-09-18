import tensorflow as tf
from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D, Dense

BATCH_SIZE = 128
VOCAB_SIZE = 90000  # 90k的词典
EMBED_FEATURE = 300 # 词向量300维
EPOCH = 1000
INIT_LEARNING_RATE = 0.0001



class TextCNN():
    def __init__(self):
        # define 3 kinds of windows size(3,4,5) for each has 100 filters

        # from shape[b, sen_len, emb_feature] => [b, conved_len, num_filters]
        self.window3_conv = Conv1D(filters=100, kernel_size=3, activation=tf.nn.relu)
        self.window4_conv = Conv1D(filters=100, kernel_size=4, activation=tf.nn.relu)
        self.window5_conv = Conv1D(filters=100, kernel_size=5, activation=tf.nn.relu)

        # from shape [b, steps, num_filters] => [b, num_filters]
        self.gobal_max_pool = GlobalMaxPool1D()
        self.fc = Dense(2, name='Dense')

    def inference(self, input, Training=None):
        # shape : [b, num_filters]
        if Training:
            input = tf.nn.dropout(input, 0.5)
        conv_w3 = self.window3_conv(input)
        conv_w4 = self.window4_conv(input)
        conv_w5 = self.window5_conv(input)

        max_feature_map_w3 = self.gobal_max_pool(conv_w3)
        max_feature_map_w4 = self.gobal_max_pool(conv_w4)
        max_feature_map_w5 = self.gobal_max_pool(conv_w5)

        # from shape : [b, num_filters] => [b, num_filters*3]
        concat_feature_map = tf.concat([max_feature_map_w3, max_feature_map_w4,
                                        max_feature_map_w5], axis=1)

        logits = self.fc(concat_feature_map)

        return logits




