import tensorflow as tf
import numpy as np
from read import generator

class single_filter_Model():
	def __init__(self, vocab_size, seq_len):
		self.batch_size = 32
		self.seq_len = seq_len
		self.classes = 2
		self.embedding_size = 100
		self.vocab_size = vocab_size
		self.placeholder()
		self.embedding()
		self.feature()
		self.fc_layer()
		self.compute_loss()



	def placeholder(self):
		self.inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
		self.labels = tf.placeholder(shape=[None, None], dtype=tf.int32)

	def embedding(self):
		embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
		_word_embeddings = tf.Variable(embeddings,
                                           dtype=tf.float32,
                                           trainable=True,
                                           name="embeddings_table")
		word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.inputs,
                                                     name="word_embeddings")
		self.embedded_chars_expanded = tf.expand_dims(word_embeddings, -1)
	
	def feature(self):
		filter_shape = [3, self.embedding_size, 1, 256]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                print('conv'+str(conv.shape))
		# Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                self.pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.seq_len - 3 + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

	def fc_layer(self):
		w = tf.get_variable(name='w',shape=[256, self.classes], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		b = tf.get_variable(name='b', initializer=tf.random_uniform([self.classes],-1.0,1.0), dtype=tf.float32)
		output = tf.reshape(self.pooled, [-1, 256])
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
			for step, (sent, label) in enumerate(batches):
				loss, _, res = sess.run([self.loss, self.train_op, self.pred],
						 	feed_dict={self.inputs:np.array(sent), self.labels:np.array(label)})
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
		for step, (sent, label) in enumerate(batches):
			#print(sent)
			#print(label)
			res = sess.run(self.pred, feed_dict={self.inputs:np.array(sent), self.labels:np.array(label)})
			for r in res:
				results.append(r)
                res = self.acc(results, labels)
                print('test_acc: '+str(res))
                return res

	def acc(self, res, labels):
		correct = 0
		num = 0.0
		for r, l in zip(res, labels):
			#print(r)
			#print(l[1])
			#print('aa')
			if r == l[1]:
				correct += 1
			num += 1
		return correct/num
