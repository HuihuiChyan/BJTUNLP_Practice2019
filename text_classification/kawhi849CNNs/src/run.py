from CNN_2345 import multi_filter_Model
from CNN_3 import single_filter_Model
from CNN_multi_layer import multi_layer_Model
from CNN_attention import attention_CNN
from CNN_2345_attention import multi_filter_attention_CNN
from transformer import Transformer
from read import read_file, read_vocab, sent2vec
import tensorflow as tf
import numpy as np


def main(_):
	tf.reset_default_graph()
	un = False
	max_len = 2048
	sents, labels = read_file('./data/train.txt')
	test_sents, test_labels = read_file('./data/test.txt')
	vocab = read_vocab()
	data = sent2vec(sents, vocab, max_len)
	test_data = sent2vec(test_sents, vocab, max_len)
	if un == True:
		un_sents, un_labels = read_file('./data/unsup.txt')
		un_data = sent2vec(un_sents, vocab, max_len)
	#model = single_filter_Model(len(vocab), max_len)
	model = multi_filter_Model(len(vocab), max_len)
	epoch = 100
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()
	sess.run(init)
	if un == True:
		model.train(un_data, un_labels, test_data, test_labels, sess, 10)
	model.train(data, labels, test_data, test_labels, sess, epoch)
	

if __name__ == '__main__':
	tf.app.run()

