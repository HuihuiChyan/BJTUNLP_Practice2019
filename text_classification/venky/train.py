# encoding=utf-8

import tensorflow as tf
import numpy as np
import data_helper
from CNN import TextCNN

# =====================定义一个训练步骤=====================

def train(cnn, data, labels, test_data, test_labels, epoches, sess, train_op):
    best_val = 0.0
    for e in range(epoches):
        Loss = 0
        results = []
        batches = data_helper.generator(zip(data, labels), cnn.batch_size)
        for step, (sent, label) in enumerate(batches):
            loss, _, res = sess.run([cnn.loss, train_op, cnn.predictions],
                                    feed_dict={cnn.inputs: np.array(sent), cnn.labels: np.array(label), cnn.dropout: 0.8})

            Loss += loss
            for r in res:
                results.append(r)
            print('epoch:' + str(e) + ' step:' + str(step) + ' loss:' + str(loss))
            if step % 100 == 0:
                val = test(test_data, test_labels, cnn.batch_size)
                if val > best_val:
                    best_val = val
                    print('Higher score:' + str(val))


def test(data, labels, batch_size):
    results = []
    batches = data_helper.generator(zip(data, labels), batch_size)
    for step, (sent, label) in enumerate(batches):
        res = sess.run(cnn.predictions,
                       feed_dict={cnn.inputs:np.array(sent), cnn.labels:np.array(label), cnn.dropout:1})
        for r in res:
            results.append(r)
    res = acc(results, labels)
    print('test_acc: ' + str(res))
    return res


def acc(res, labels):
    correct = 0
    num = 0.0
    for r, l in zip(res, labels):
        if r == l[1]:
            correct += 1
        num += 1
    return correct / num


# =====================数据预处理=====================

# 加载数据，返回数据集和标签
print("Loading data...")
sents, labels = data_helper.load_data('./data/train.txt')
test_sents, test_labels = data_helper.load_data('./data/test.txt')

max_len = 1024
vocab = data_helper.read_vocab()
data = data_helper.sent2idx(sents, vocab, max_len)
test_data = data_helper.sent2idx(test_sents, vocab, max_len)
epoch = 100


with tf.Graph().as_default():
    session_conf = tf.ConfigProto()
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(vocab_size=len(vocab),
                      seq_len=max_len,
                      embedding_size=256,
                      num_classes=2,
                      filter_sizes=[3, 4, 5],
                      num_filters=256)


        # 指定优化器（梯度下降）
        optimizer = tf.train.AdamOptimizer(1e-3)
        # 梯度
        train_op = optimizer.minimize(cnn.loss)


        # 全局初始化 Initialize all variables
        sess.run(tf.global_variables_initializer())
        train(cnn, data, labels, test_data, test_labels, epoch, sess, train_op)