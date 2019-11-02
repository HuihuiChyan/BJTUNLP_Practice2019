import os
import math
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import generator, sent2vec
from eval import conlleval
class Model():
	def __init__(self, args):
		
		self.lr = 0.001
		self.vocab_size = args['vocab_size']
		self.classes = args['classes']
		self.hidden_dim = args['hidden_dim']
		self.vocab_t = args['vocab_t']
		self.input_x = tf.placeholder(shape=[None, None], dtype=tf.int32)
		self.input_y = tf.placeholder(shape=[None, None], dtype=tf.int32)
		self.seq_len = tf.placeholder(shape=[None], dtype=tf.int32)
		self.dropout = tf.placeholder(dtype = tf.float32)
		self.max_len = args['max_len']
		self.batch_size = args['batch_size']
		self.model_name = args['model_name']
		self.pretrained = False
                if self.pretrained == True:
                        embed = np.load('./embedding/vocab.npy')
                        trainable = False
                else:
                        embed = tf.random_uniform([self.vocab_size, self.hidden_dim], -1.0, 1.0)
                        trainable = True
                _word_embeddings = tf.Variable(embed, trainable=trainable, dtype=tf.float32)
                self.embed = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.input_x)
                #dropout
		self.embed = tf.nn.dropout(self.embed, 1-self.dropout)
		if args['encoder'] == 'BiLSTM':
			self.feature = self.BiLSTM()
#			self.feature = self.atten()
		else:
			self.feature = embed
                w = tf.get_variable(name='w', shape=[2 * self.hidden_dim, self.classes], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b = tf.get_variable(name='b', shape=[self.classes], initializer=tf.zeros_initializer(), dtype=tf.float32)
                out = tf.matmul(tf.reshape(self.feature, [-1, 2 * self.hidden_dim]), w) + b
                self.logits = tf.reshape(out, [-1, self.max_len, self.classes])
		
		log_likelihood, self.transition_params = crf_log_likelihood(
				inputs=self.logits, tag_indices=self.input_y, sequence_lengths=self.seq_len)
		self.loss = -tf.reduce_mean(log_likelihood)
		
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		optm = tf.train.AdamOptimizer(learning_rate=self.lr)
		grads_and_vars = optm.compute_gradients(self.loss)
		grads_and_vars_clip = [[tf.clip_by_value(g, -5.0, 5.0), v] for g, v in grads_and_vars]
		self.train_op = optm.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

	def BiLSTM(self):
		cell_fw, cell_bw = LSTMCell(self.hidden_dim), LSTMCell(self.hidden_dim)
		(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
			cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.embed, sequence_length=self.seq_len, dtype=tf.float32)
		output = tf.concat([output_fw, output_bw], -1)
		#dropout
		output = tf.nn.dropout(output, 1-self.dropout)
		return output

	def atten(self):
		with tf.name_scope("self_atten"):
			embed_size = self.hidden_dim*2
			word_embs = self.feature
			seq_len = self.max_len#2048#self.sequence_lengths
			w_q = tf.get_variable(name='w_q', shape=[embed_size, embed_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
			w_k = tf.get_variable(name='w_k', shape=[embed_size, embed_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
			w_v = tf.get_variable(name='w_v', shape=[embed_size, embed_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
			Q = tf.matmul(tf.reshape(word_embs, shape=[-1, embed_size]), w_q)
			K = tf.matmul(tf.reshape(word_embs, shape=[-1, embed_size]), w_k)
			V = tf.matmul(tf.reshape(word_embs, shape=[-1, embed_size]), w_v)
			Q = tf.reshape(Q, shape=[-1, seq_len, embed_size])
			K = tf.reshape(K, shape=[-1, seq_len, embed_size])
			V = tf.reshape(V, shape=[-1, seq_len, embed_size])
			K = tf.transpose(K, perm=[0, 2, 1])
			self.attention = tf.nn.softmax(tf.matmul(Q, K))
			self.atten_output = tf.matmul(self.attention/math.sqrt(embed_size), V)
		return self.atten_output

	def train(self, sess, train_sent, train_len, train_label, test_sent, test_len, test_label, raw_data):
		epoch = 50
		for e in range(epoch):
			batches = generator(zip(train_sent, train_len, train_label), self.batch_size)
			for step, (x, seq_len, y) in enumerate(batches):
				_, loss = sess.run([self.train_op, self.loss], 
						feed_dict={self.input_x:x,self.input_y:y, self.seq_len:seq_len, self.dropout:0.3})
				print('epoch '+str(e)+', step '+str(step)+', loss '+str(loss))	
				
				if step % 40 == 0:
					self.test(sess, test_sent, test_len, test_label, e, step, raw_data)

	def test(self, sess, sents, lens, labels, epoch, step, raw_data):
		batches = generator(zip(sents, lens, labels), self.batch_size)
		res, seq_lens = [], []
		for x, seq_len, y in batches:
			logits, transition_params = sess.run([self.logits, self.transition_params], 
						feed_dict={self.input_x:x, self.input_y:y, self.seq_len:seq_len, self.dropout:0})
			seq_lens.extend(seq_len)
			for logit, seq_l in zip(logits, seq_len):
				viterbi_seq, _ = viterbi_decode(logit[:seq_l], transition_params)
				res.append(viterbi_seq)
		self.evaluate(res, seq_lens, zip(sents, labels), epoch, step, raw_data)
	
	def evaluate(self, labels, seq_lens, data, epoch, step, raw_data):
		label2tag = {}
		for tag, label in self.vocab_t.items():
			label2tag[label] = tag# if label != 0 else label
		prediction = []
		for label, sent, tag in zip(labels, raw_data['test_data'], raw_data['test_label']):
			tag_ = [label2tag[l] for l in label]
			res = []
			for i in range(len(tag_)):
				res.append([sent[i], tag[i], tag_[i]])
			prediction.append(res)
		print conlleval(prediction)

		
