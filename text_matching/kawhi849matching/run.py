from model import Model
from read import read_file, read_vocab, sent2vec
import tensorflow as tf
import numpy as np


def main(_):
#	tf.reset_default_graph()
	un = False
	max_len = 128
	sents, labels = read_file('./data/train')
	test_sents, test_labels = read_file('./data/test')
	vocab = read_vocab()
	data = sent2vec(sents, vocab, max_len)
	test_data = sent2vec(test_sents, vocab, max_len)
	model = Model(len(vocab), max_len)
	#model = Transformer(len(vocab), max_len)
	epoch = 100
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()
	sess.run(init)
	model.train(data, labels, test_data, test_labels, sess, epoch)
	

if __name__ == '__main__':
	tf.app.run()

