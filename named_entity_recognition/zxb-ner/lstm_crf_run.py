# coding:utf-8

import os, time, sys
import logging, sys, argparse, random
import tensorflow as tf
import numpy as np
from eval import conlleval


def exec():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO,
                        datefmt='%I:%M:%S')
    parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
    parser.add_argument('--train_data', type=str, default='/home1/huanghui/dataset4practice/named_entity_recognition/conll2003/', help='train data source')
    parser.add_argument('--test_data', type=str, default='/home1/huanghui/dataset4practice/named_entity_recognition/conll2003/', help='test data source')
    #parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
    #parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
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
    output_path = os.path.join('./', "_save", timestamp)
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

    train_path_src = os.path.join('../..', args.train_data, 'eng.train.src')
    train_path_trg = os.path.join('../..', args.train_data, 'eng.train.trg')

    test_path_src = os.path.join('../..', args.test_data, 'eng.testb.src')
    test_path_trg = os.path.join('../..', args.test_data, 'eng.testb.trg')

    #train_data = read_corpus(train_path_src, train_path_trg)
    #test_data = read_corpus(test_path_src, test_path_trg)

    train_data,train_labels = read_file(train_path_src, train_path_trg)
    test_data,test_labels = read_file(test_path_src, test_path_trg)

    vocab_path = os.path.join('../..', args.train_data, 'vocab.w')
    vocab_t_path = os.path.join('../..', args.train_data, 'vocab.t')

    # 构建词表
    vocab, vocab_t = read_vocab(vocab_path, vocab_t_path)

    train_max_len = max(map(lambda x: len(x), train_data))
    test_max_len = max(map(lambda x: len(x), test_data))
    max_length = max(train_max_len,test_max_len)

    train_vecs,train_lens,train_tags,train_raw_data,train_raw_tag = sent2vec(train_data,train_labels,vocab,vocab_t,max_length)
    test_data, test_lens, test_tags,test_raw_data,test_raw_tag = sent2vec(test_data, test_labels, vocab, vocab_t, max_length)

    raw_data = dict()
    raw_data['test_data'] = test_raw_data
    raw_data['test_label'] = test_raw_tag

    vocab_size = len(vocab)

    # 这里是下载下来的bert配置文件

    embeddings = random_embedding(vocab_size=vocab_size, embedding_dim=args.embedding_dim)

    if args.mode == "train":
        model = BiLSTM_CRF(args, embeddings, vocab_t, vocab_size, paths)
        model.build_graph()
        logging.info("train data:{}".format(len(train_data)))
        # print("train data:{}".format(len(train_data)))
        model.train(train_vecs,train_lens,train_tags, test_data, test_lens, test_tags, config,raw_data)


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings,tag2label, vocab_size, paths):
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
            attn_length = 20
            lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            #lstm_fw = tf.contrib.rnn.AttentionCellWrapper(lstm_fw, attn_length, state_is_tuple=True)
            lstm_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            #lstm_bw = tf.contrib.rnn.AttentionCellWrapper(lstm_bw, attn_length, state_is_tuple=True)
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

            outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
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

    def train(self, train_vecs,train_lens,train_tags, test_data, test_lens, test_tags, config,raw_data):

        with tf.Session(config=config) as sess:
            sess.run(self.init_op)
            #self.epoch_num = 5
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train_vecs,train_lens,train_tags,test_data, test_lens, test_tags, epoch,raw_data)

    def build_graph(self):
        self.add_placeholder()
        self.lookup_layer_op()
        self.Bilstm_layer_op()
        #self.Lstm_layer_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def run_one_epoch(self, sess, train_vecs,train_lens,train_tags, test_data, test_lens, test_tags, epoch,raw_data):
        # print("1################1")
        # print("++++epoch+++", epoch)
        logging.info("++++epoch+++" + str(epoch))
        #batches = batch_yield(train, self.batch_size, vocab_index_dict, self.tag2label, shuffle=False)
        trainBatches = generator(zip(train_vecs,train_lens,train_tags), self.batch_size)

        for step, (seqs,seq_lens,labels) in enumerate(trainBatches):
            feed_dict, _ = self.get_feed_dict(seqs,seq_lens,labels, self.lr, self.dropout_keep_prob)
            _, loss_train, step_num_ = sess.run([self.train_op, self.loss, self.global_step], feed_dict=feed_dict)
            #_, train_loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            # print("loss_train:%.3f%%" % train_loss)

        # print('===========validation / test===========')
        logging.info('===========validation / test===========')
        label_list_test, label_list_test_len = [], []
        testBatches = generator(zip(test_data, test_lens, test_tags), self.batch_size)
        for step, (seqs,seq_lens,labels) in enumerate(testBatches):
            label_list, seq_len_list = self.predict_one_batch(sess, seqs,seq_lens,labels)
            label_list_test.extend(label_list)
            label_list_test_len.extend(seq_len_list)
            # print("###predict###",label_list,"lentgth:###",seq_len_list)

        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, sent_, tag in zip(label_list_test, raw_data['test_data'], raw_data['test_label']):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            #logging.info('len(label_):%.3f%%' % len(label_))
            #logging.info("len(sent_):%.3f%%" % len(sent_))
            #logging.info("len(tag):%.3f%%" % len(tag))


            for i in range(len(sent_)):
                #tagStr = 'sent_:' + sent_[i] + 'tag[i]:' + tag[i] + 'tag_[i]:' + tag_[i]
                #logging.info("tagStr:%s" % tagStr)
                sent_res.append([sent_[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch + 1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)

        for _ in conlleval(model_predict, label_path, metric_path):
            logging.info(_)

    def predict_one_batch(self, sess, seqs,seq_lens,labels):
        feed_dict, seq_len_list = self.get_feed_dict(seqs=seqs,seq_lens=seq_lens,labels= labels,lr=None,dropout=1.0)
        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
            label_list = []
            for logits, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logits[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

    def get_feed_dict(self,seqs,seq_lens,labels=None, lr=None, dropout=None):
        seq_len_list =[]
        feed_dict = {
            self.words_ids: seqs,
            self.sequence_lengths: seq_lens,
            self.labels: labels
        }
        seq_len_list.extend(seq_lens)
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


def generator(data,batch_size):
    seqs,seq_lens,labels=[],[],[]
    for sent,seq_len,label in data:
        if len(seqs) == batch_size:
            yield seqs,seq_lens,labels
            seqs, seq_lens, labels =[],[],[]
        seqs.append(sent)
        seq_lens.append(seq_len)
        labels.append(label)

    if len(seqs) != 0:
        yield seqs, seq_lens, labels

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


def read_corpus(train_path_src, train_path_trg):
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
    for i, char in enumerate(sent_):
        char_tag = tag_[i]
        if char == "sep" and char_tag == "sep":
            count = count + 1
        else:
            article.append(char)
            article_tag.append(char_tag)
            if count != 0 and count % 10 == 0:
                count = count + 1
                data.append((article, article_tag))
                article, article_tag = [], []

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
    embedding_mat = np.random.uniform(-1.0, 1.0, (vocab_size, embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def read_vocab(data_path, label_path):
    vocab = dict()
    vocab_t = dict()
    num = 0
    for line in open(data_path):
        vocab[line.strip()] = num
        num += 1
    t_num = 0
    for t_line in open(label_path):
        #vocab_t[t_line.strip()] = t_num
        #t_num += 1
        if t_line =='<eos>\n':
            continue
        else:
            vocab_t[t_line.strip()] = t_num
            t_num += 1
    return vocab, vocab_t


def read_file(data_path, label_path):
    texts = []
    labels = []
    for text_line in open(data_path):
        texts.append(text_line.strip().split(' '))
    for label_line in open(label_path):
        labels.append(label_line.strip().split(' '))
    return texts,labels

def sent2vec(sents,labels,vocab,vocab_t,max_len):
    #向量和长度
    vecs =[]
    lens=[]
    raw_data=[]
    for sent in sents:
        vec = [vocab[word] if word in vocab else 0 for word in sent]
        length = len(vec)
        if length <=max_len:
            vec +=[0 for _ in range(max_len-length)]
            raw_sent = sent
        else:
            vec = vec[:max_len]
            raw_sent =sent[:max_len]

        vecs.append(vec)
        addLength = min(max_len,length)
        lens.append(addLength)
        raw_data.append(raw_sent)
    #标签
    tags =[]
    raw_tag = []
    for label in labels:
        tag = [vocab_t[l] for l in label]
        if len(tag) <=max_len:
            tag +=[0 for _ in range(max_len-len(tag))]
            tag_raw = label
        else:
            tag = tag[:max_len]
            tag_raw = label[:max_len]
        tags.append(tag)
        raw_tag.append(tag_raw)

    return vecs,lens,tags,raw_data,raw_tag


if __name__ == '__main__':
    exec()
