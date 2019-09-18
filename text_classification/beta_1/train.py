import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import data_helper
import gensim
import TextCNN
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

TRAIN_DATASET_DIR = 'dataset/train_dataset'     # directory of train dataset
TEST_DATASET_DIR = 'dataset/test_dataset'       # directory of test dataset

x_train, y_train = data_helper.load_data_and_labels_new('train.txt')
x_test, y_test = data_helper.load_data_and_labels_new('test.txt')
embedding_table = np.load('dataset/imdb.w2v.npy')
# print(embedding_table[:5])


def train():
    x = tf.placeholder(tf.int32, [None, None], name='x')
    y = tf.placeholder(tf.int32, [None], name='y')
    lr = TextCNN.INIT_LEARNING_RATE
    embedding = tf.Variable(embedding_table, dtype=tf.float32, trainable=False)
    # embedding = tf.Variable(tf.random_uniform([TextCNN.VOCAB_SIZE, TextCNN.EMBED_FEATURE], -1.0, 1.0))
    input = tf.nn.embedding_lookup(embedding, x)

    model = TextCNN.TextCNN()
    logits_train = model.inference(input, Training=True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=y, name='loss')
    loss_ = tf.reduce_mean(loss) + tf.nn.l2_loss(model.fc.get_weights()[0])
    train_op = tf.train.AdamOptimizer(lr).minimize(loss_)

    logits = model.inference(input)
    correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.cast(y, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    sum_correct_pred = tf.reduce_sum(tf.cast(correct_pred, dtype=tf.float32))



    with tf.Session() as sess:
        max_score = 0
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for epoch in range(TextCNN.EPOCH):
            for step, (x_, y_) in enumerate(data_helper.batch_iter(x_train, y_train, TextCNN.BATCH_SIZE)):
                # print(sess.run(input , feed_dict={x:x_}))
                _ = sess.run(train_op, feed_dict={x: x_, y: y_})
                # if step % 64 == 0:
                    # print(y_)
                    # print(sess.run(tf.argmax(logits, axis=1), feed_dict={x: x_}))
                    # print('epoch :', epoch, 'step :', step, ' train_acc = ', sess.run(accuracy, feed_dict={x: x_, y: y_}))
            sum_ = 0
            for (x__, y__) in data_helper.batch_iter(x_test, y_test, TextCNN.BATCH_SIZE):
                tmp = sess.run(sum_correct_pred, feed_dict={x: x__, y: y__})
                sum_ += tmp
            print('epoch ', epoch, 'acc = ', sum_/len(y_test))
            max_score = max(sum_/len(y_test), max_score)
            if epoch % 30 == 0:
                lr /= 2
        print('Best Accuracy: %.4f' % max_score)

if __name__ == '__main__':
    train()






