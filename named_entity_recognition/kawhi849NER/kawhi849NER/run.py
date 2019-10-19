import sys
import tensorflow as tf
import numpy as np
from model import Model
from data import read_file, read_vocab, sent2vec



def main(_):
	max_len = 64
	args = dict()
	args['encoder'] = 'BiLSTM'
	train_texts, train_labels = read_file('./conll2003/eng.train.src', './conll2003/eng.train.trg')
	test_texts, test_labels = read_file('./conll2003/eng.testa.src', './conll2003/eng.testa.trg')
	vocab, vocab_t = read_vocab('./conll2003/vocab.w', './conll2003/vocab.t')
	train_vecs, train_lens, train_tags = sent2vec(train_texts, train_labels, vocab, vocab_t, max_len)
	test_vecs, test_lens, test_tags = sent2vec(test_texts, test_labels, vocab, vocab_t, max_len)

	args = dict()
	args['model_name'] = sys.argv[1]
	args['encoder'] = 'BiLSTM'
	args['vocab_size'] = len(vocab)
	args['classes'] = len(vocab_t)
	args['hidden_dim'] = 300
	args['vocab_t'] = vocab_t
	args['max_len'] = max_len
	args['batch_size'] = 32
	raw_data = dict()
	raw_data['test_data'] = test_texts
	raw_data['test_label'] = test_labels
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	model = Model(args)
	sess.run(tf.global_variables_initializer())
	model.train(sess, train_vecs, train_lens, train_tags, test_vecs, test_lens, test_tags, raw_data)
	

if __name__ == '__main__':
	tf.app.run()
