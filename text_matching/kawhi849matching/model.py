import tensorflow as tf
import numpy as np
from read import generator

class Model():
	def __init__(self, vocab_size, seq_len):
		self.batch_size = 32
		self.seq_len = seq_len
		self.classes = 3
		self.embedding_size = 300
		self.vocab_size = vocab_size
		self.pretrain_embedding = False #True
		self.placeholder()
		self.embedding()
		self.feature()
		self.fc_layer()
		self.compute_loss()


	def placeholder(self):
		self.input_a = tf.placeholder(shape=[None, None], dtype=tf.int32)
		self.input_b = tf.placeholder(shape=[None, None], dtype=tf.int32)
		self.labels = tf.placeholder(shape=[None, None], dtype=tf.float32)
		self.dropout = tf.placeholder(dtype=tf.float32)
	def embedding(self):
		if self.pretrain_embedding == True:
			embeddings = np.load('./embedding/vocab.npy')
			trainable=False
		else:
			embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
			trainable = True
		_word_embeddings = tf.Variable(embeddings,
                                           dtype=tf.float32,
                                           trainable=trainable,
                                           name="embeddings_table")
		word_embeddings_a = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.input_a,
                                                     name="word_embeddings_1")
		word_embeddings_b = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.input_b,
                                                     name="word_embeddings_2")
		self.embedded_chars_expanded_a = tf.expand_dims(word_embeddings_a, -1)
		self.embedded_chars_expanded_b = tf.expand_dims(word_embeddings_b, -1)
		self.embedded_chars_expanded_a = tf.nn.dropout(self.embedded_chars_expanded_a, self.dropout)
		self.embedded_chars_expanded_b = tf.nn.dropout(self.embedded_chars_expanded_b, self.dropout)


	def feature(self):
		outputs = []
		self.filter_list = [2, 3, 4, 5]
		for filter_size in self.filter_list:
			filter_shape = [filter_size, self.embedding_size, 1, 256]
                	W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                	b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                	conv = tf.nn.conv2d(
                    		self.embedded_chars_expanded_a,
                    		W,
                    		strides=[1, 1, 1, 1],
                    		padding="VALID",
                    		name="conv_%d" % filter_size)
                	h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu_%d" % filter_size)
                	self.pooled = tf.nn.max_pool(
                    		h,
                    		ksize=[1, self.seq_len - filter_size + 1, 1, 1],
                    		strides=[1, 1, 1, 1],
                    		padding='VALID',
                    		name="pool_%d" % filter_size)
			outputs.append(self.pooled)
		self.pooled_output = tf.concat(outputs, 3) # concat on last dimension
		self.feature_a = tf.reshape(self.pooled_output, [-1, len(self.filter_list)*256])
		outputs = []
		for filter_size in self.filter_list:
			filter_shape = [filter_size, self.embedding_size, 1, 256]
                	W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                	b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                	conv = tf.nn.conv2d(
                    		self.embedded_chars_expanded_b,
                    		W,
                    		strides=[1, 1, 1, 1],
                    		padding="VALID",
                    		name="conv_%d" % filter_size)
                	h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu_%d" % filter_size)
                	self.pooled = tf.nn.max_pool(
                    		h,
                    		ksize=[1, self.seq_len - filter_size + 1, 1, 1],
                    		strides=[1, 1, 1, 1],
                    		padding='VALID',
                    		name="pool_%d" % filter_size)
			outputs.append(self.pooled)
		self.pooled_output = tf.concat(outputs, 3) # concat on last dimension
		self.feature_b = tf.reshape(self.pooled_output, [-1, len(self.filter_list)*256])
		print(self.feature_b.shape)

	def fc_layer(self):
		self.features = tf.abs(tf.reduce_mean(self.feature_a, axis=-1) - tf.reduce_mean(self.feature_b, axis=-1))
		#w = tf.get_variable(name='w',shape=[len(self.filter_list) * 256, self.classes], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		print(self.features.get_shape().as_list()[0])
		w = tf.get_variable(name='w',shape=[self.features.get_shape().as_list()[0], self.classes], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		b = tf.get_variable(name='b', initializer=tf.random_uniform([self.classes],-1.0,1.0), dtype=tf.float32)
		output = tf.reshape(self.features, [-1, self.features.get_shape().as_list()[0]])
		self.logits = tf.matmul(output,w) + b

	def compute_loss(self):
		global_step = tf.Variable(0, trainable=False)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
		self.train_op = self.optimizer.minimize(self.loss)
		self.pred = tf.argmax(self.logits, 1)

	def train(self, data, labels, test_data, test_labels, sess, epoches):
		best_val = 0.0
		for e in range(epoches):
			Loss = 0
			results = []
			batches = generator(zip(data, labels), self.batch_size)
			for step, (sent_a, sent_b, label) in enumerate(batches):
				loss, _, res = sess.run([self.loss, self.train_op, self.pred],
						 	feed_dict={self.input_a:np.array(sent_a), self.input_b:np.array(sent_b), self.labels:np.array(label), self.dropout:0.8})
				Loss += loss
				for r in res:
					results.append(r)
				print('epoch: ' + str(e) +' step: ' + str(step) +' loss: '+str(loss))
				if step % 100 == 0:
					val = self.test(test_data, test_labels, sess)
					if val > best_val:
						best_val = val
						print('Higher score, '+ str(val))


	def test(self, data, labels, sess):
		results = []
		batches = generator(zip(data, labels), self.batch_size)
		for step, (sent_a, sent_b, label) in enumerate(batches):
			res = sess.run(self.pred, feed_dict={self.input_a:np.array(sent_a), self.input_b:np.array(sent_b),self.labels:np.array(label), self.dropout:1})
			for r in res:
				results.append(r)
		res = self.acc(results, labels)
		print('test_acc: '+str(res))
		return res

	def acc(self, res, labels):
		correct = 0
		num = 0.0
		for r, l in zip(res, labels):
			if r == l[1]:
				correct += 1
			num += 1
		return correct/num
