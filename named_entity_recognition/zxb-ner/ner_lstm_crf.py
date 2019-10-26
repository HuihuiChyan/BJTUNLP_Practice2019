# coding:utf-8

import os, time, sys
import logging, sys, argparse, random
import tensorflow as tf
import numpy as np
from eval import conlleval
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

def exec():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO,
                        datefmt='%I:%M:%S')
    parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')

    parser.add_argument('--train_data', type=str, default='/home1/huanghui/dataset4practice/named_entity_recognition/conll2003/', help='train data source')
    parser.add_argument('--test_data', type=str, default='/home1/huanghui/dataset4practice/named_entity_recognition/conll2003/', help='test data source')
    parser.add_argument('--batch_size', type=int, default=10, help='#sample of each minibatch')
    parser.add_argument('--epoch', type=int, default=3, help='#epoch of training')
    parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
    parser.add_argument('--CRF', type=bool, default=True, help='use CRF at the top layer. if False, use Softmax')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
    parser.add_argument('--update_embedding', type=bool, default=True, help='update embedding during training')
    parser.add_argument('--pretrain_embedding', type=str, default='random',
                        help='use pretrained char embedding or init it randomly')
    parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle training data before each epoch')
    parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
    parser.add_argument('--demo_model', type=str, default='1548506006', help='model for test and demo')
    args = parser.parse_args()

    paths = {}
    
    timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
    output_path = os.path.join('.', "./model/" + "_save", timestamp)
    if not os.path.exists(output_path): os.makedirs(output_path)
    summary_path = os.path.join(output_path, "summaries")
    paths['summary_path'] = summary_path
    if not os.path.exists(summary_path): os.makedirs(summary_path)
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path): os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    paths['model_path'] = ckpt_prefix
    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path): os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path
  
    tag2label = {
        "O": 0,
        "S-PER": 1,
        "B-PER": 2,
        "I-PER": 3,
        "E-PER": 4,

        "S-LOC": 5,
        "B-LOC": 6,
        "I-LOC": 7,
        "E-LOC": 8,

        "S-ORG": 9,
        "B-ORG": 10,
        "I-ORG": 11,
        "E-ORG": 12,

        "S-MISC": 13,
        "B-MISC": 14,
        "I-MISC": 15,
        "E-MISC": 16
    }

    train_path_src = os.path.join('/', args.train_data, 'eng.train.src')
    train_path_trg = os.path.join('/', args.train_data, 'eng.train.trg')


    test_path_src = os.path.join('/', args.test_data, 'eng.testb.src')
    test_path_trg = os.path.join('/', args.test_data, 'eng.testb.trg')

    train_data = read_corpus(train_path_src,train_path_trg)
    test_data = read_corpus(test_path_src,test_path_trg)

    dict = train_data + test_data

    vocab_index_dict, index_vocab_dict, vocab_size = create_vocab_dict(dict)

    # 这里是下载下来的bert配置文件

    embeddings = random_embedding(vocab_size=vocab_size, embedding_dim=args.embedding_dim)

    if args.mode == "train":
        model = BiLSTM_CRF(args, embeddings, tag2label, vocab_index_dict, index_vocab_dict, vocab_size, paths)
        model.build_graph()
        logging.info("train data:{}".format(len(train_data)))
        #print("train data:{}".format(len(train_data)))
        model.train(train_data, test_data, vocab_index_dict,config)


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab_index_dict, index_vocab_dict, vocab_size, paths):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab_size = vocab_size
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.result_path = paths['result_path']

    def add_placeholder(self):
        self.words_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="words_ids")
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(
                self.embeddings,
                dtype=tf.float32,
                trainable=self.update_embedding,
                name="_word_embedding"
            )
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.words_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def Bilstm_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            attn_length = 120
            lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            lstm_fw = tf.contrib.rnn.AttentionCellWrapper(lstm_fw, attn_length, state_is_tuple=True)
            lstm_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            lstm_bw = tf.contrib.rnn.AttentionCellWrapper(lstm_bw, attn_length, state_is_tuple=True)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw,
                cell_bw=lstm_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32
            )
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W", shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable(name="b", shape=[self.num_tags], initializer=tf.zeros_initializer(), dtype=tf.float32)
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def Lstm_layer_op(self):
        with tf.variable_scope("lstm"):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim)

            outputs,_ = tf.nn.dynamic_rnn(cell = lstm_cell,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            output = tf.nn.dropout(outputs, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W", shape=[self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable(name="b", shape=[self.num_tags], initializer=tf.zeros_initializer(), dtype=tf.float32)
            s = tf.shape(output)
            output = tf.reshape(output, [-1, self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs=self.logits,
                tag_indices=self.labels,
                sequence_lengths=self.sequence_lengths
            )
            self.loss = tf.reduce_mean(-log_likelihood)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            mean_loss = tf.reduce_mean(self.loss)
            self.train_op = optim.minimize(mean_loss)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def train(self, train_data, test_data, vocab_index_dict,config):
        with tf.Session(config=config) as sess:
            sess.run(self.init_op)
            #self.epoch_num = 5
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train_data, test_data, epoch, vocab_index_dict)

    def build_graph(self):
        self.add_placeholder()
        self.lookup_layer_op()
        #self.Bilstm_layer_op()
        self.Lstm_layer_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def run_one_epoch(self, sess, train, test, epoch, vocab_index_dict):
        # print("1################1")
        #print("++++epoch+++", epoch)
        logging.info("++++epoch+++"+str(epoch))
        batches = batch_yield(train, self.batch_size, vocab_index_dict, self.tag2label, shuffle=False)

        for step, (seqs, labels) in enumerate(batches):
            # print("2################", sent_, "***", tag_)
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, step_num_ = sess.run([self.train_op, self.loss, self.global_step], feed_dict=feed_dict)
            #print("loss_train:%.3f%%" % loss_train)

        #print('===========validation / test===========')
        logging.info('===========validation / test===========')
        label_list_test, label_list_test_len = [], []
        for seqs, labels in batch_yield(test, self.batch_size, vocab_index_dict, self.tag2label, shuffle=False):
            label_list, seq_len_list = self.predict_one_batch(sess, seqs)
            label_list_test.extend(label_list)
            label_list_test_len.extend(seq_len_list)
            # print("###predict###",label_list,"lentgth:###",seq_len_list)

        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
            #label2tag[label] = tag
        #print("+++")
        model_predict =[]
        for label_,(sent_,tag) in zip(label_list_test,test):
            #tag_ = [label2tag[label_2]  for label_2 in label_]
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            for i in range(len(sent_)):
                sent_res.append([sent_[i],tag[i],tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch + 1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)

        for _ in conlleval(model_predict,label_path,metric_path):
            logging.info(_)
            #print(_)

    def predict_one_batch(self, sess, seqs):
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)
        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
            label_list = []
            for logits, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logits[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        feed_dict = {
            self.words_ids: word_ids,
            self.sequence_lengths: seq_len_list
        }
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        return feed_dict, seq_len_list


def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_list_len = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_list_len.append(min(len(seq_), max_len))
    return seq_list, seq_list_len


def batch_yield(data, batch_size, vocab_index_dict, tag2label, shuffle=False):
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ids = sentence2id(sent_, vocab_index_dict)
        label_ids = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_ids)
        labels.append(label_ids)

    if len(seqs) != 0:
        yield seqs, labels


def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        sentence_id.append(word2id[word])
    return sentence_id


def read_corpus(train_path_src,train_path_trg):
    data = []
    with open(train_path_src, encoding="utf-8") as fr:
        lines1 = fr.readlines()

    sent_, tag_ = [], []
    for l1 in lines1:
        if l1 != "\n":
            str = l1.strip().split()
            sent_.extend(str)
            sent_.append("sep")

    with open(train_path_trg, encoding="utf-8") as fr:
        lines2 = fr.readlines()

    for l2 in lines2:
        if l2 != "\n":
            tag = l2.strip().split()
            tag_.extend(tag)
            tag_.append("sep")

    article = []
    article_tag = []
    count = 0
    for i,char in enumerate(sent_):
        char_tag = tag_[i]
        if char == "sep" and char_tag == "sep":
            #count = count+1
            data.append((article, article_tag))
            article,article_tag = [],[]
        else:
            article.append(char)
            article_tag.append(char_tag)
            #if count!=0 and count%10 ==0:
                #count = count+1
            #    data.append((article, article_tag))
            #    article,article_tag = [],[]

    return data


def create_vocab_dict(data):
    vocab_index_dict = {}
    index_vocab_dict = {}

    array = []

    for (sent_, label) in data:
        for word in sent_:
            array.append(word)

    unique_chars = set(array)
    vocab_size = len(unique_chars)

    for i, word in enumerate(unique_chars):
        vocab_index_dict[word] = i
        index_vocab_dict[i] = word

    return vocab_index_dict, index_vocab_dict, vocab_size


def random_embedding(vocab_size, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


if __name__ == '__main__':
    exec()
