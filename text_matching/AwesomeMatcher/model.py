import tensorflow as tf
import pdb

class InferSent(object):

	def __init__(self, config):
		"""
		config - dict: contains parameters
		"""
		self.maxlen = config['maxlen']
		self.hidden_size = config['hidden_size']
		self.learning_rate = config['lr']
		self.vocab_size = config['vocab_size']
		self.dropout = config['dropout']
		self.kernel_size = config['kernel']
		self.class_size = config['class_size']
		self.batch_size = config['batch_size']
		
		self.init_placeholder()
		self.build_graph()

	def init_placeholder(self):
		self.input1 = tf.placeholder(dtype=tf.int32, shape=[None, self.maxlen], name='input1')
		self.input2 = tf.placeholder(dtype=tf.int32, shape=[None, self.maxlen], name='input2')
		self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')

	def build_graph(self):
		# sentA sentB
		#[batch_size, maxlen]
		self.embedding = tf.get_variable(dtype=tf.float32, 
										 initializer=tf.random_uniform([self.vocab_size, self.hidden_size], -1., 1.),
										 name='embedding')
		input_vec_a = tf.nn.embedding_lookup(self.embedding, self.input1)
		input_vec_b = tf.nn.embedding_lookup(self.embedding, self.input2)

		res = self.build_mid_layer_multi_cnn(input_vec_a, input_vec_b)

		self.loss = self.loss_op(res)

		optim = tf.train.AdamOptimizer(self.learning_rate)

		self.train_op = optim.minimize(self.loss)

		self.logit = tf.nn.softmax(res, axis=-1)

		self.pred_y = tf.argmax(self.logit, axis=-1)

		self.acc = tf.reduce_mean(tf.cast(tf.equal(self.label, tf.cast(self.pred_y, dtype=tf.int32)), dtype=tf.float32))

	def build_mid_layer_multi_cnn(self, input_vec_a, input_vec_b, norm='none'):

		input_vec_a = tf.expand_dims(input_vec_a, axis=-1)
		input_vec_b = tf.expand_dims(input_vec_b, axis=-1)
		#[batch_size, maxlen, hidden_size, 1]
		
		conv_vec_as = []
		conv_vec_bs = []

		for i in range(len(self.kernel_size)):
			with tf.variable_scope("conv_%d" % (i)):
				conv_vec_a = tf.layers.conv2d(input_vec_a,
											  filters=self.hidden_size,
											  kernel_size=[self.kernel_size[i], self.hidden_size],
											  activation=tf.nn.relu,
											  kernel_initializer=tf.contrib.layers.xavier_initializer(),
											  name='encoder-1',
											  padding='VALID')
				conv_vec_b = tf.layers.conv2d(input_vec_b,
											  filters=self.hidden_size,
											  kernel_size=[self.kernel_size[i], self.hidden_size],
											  activation=tf.nn.relu,
											  kernel_initializer=tf.contrib.layers.xavier_initializer(),
											  name='encoder-2',
											  padding='VALID')
				if norm=="batch_norm":
					conv_vec_a = tf.contrib.layers.batch_norm(conv_vec_a,
															  scope='batchnorm1',
															  updates_collections=None)
					conv_vec_b = tf.contrib.layers.batch_norm(conv_vec_b,
															  scope='batchnorm2',
															  updates_collections=None)
				elif norm=="layer_norm":
					conv_vec_a = tf.contrib.layers.layer_norm(conv_vec_a,
															  scope='layernorm1',
															  begin_norm_axis=-1)
					conv_vec_b = tf.contrib.layers.layer_norm(conv_vec_b,
															  scope='layernorm2',
															  begin_norm_axis=-1)

				elif norm=="none":
					pass

				else:
					raise Exception("Unknown normalization function.")

				#[batch_size, maxlen-kernel_size, 1, output_channel]
				pool_vec_a = tf.reduce_mean(conv_vec_a, axis=-3)
				pool_vec_b = tf.reduce_mean(conv_vec_b, axis=-3)
				#[batch_size, 1, output_channel]
				conv_vec_as.append(pool_vec_a)
				conv_vec_bs.append(pool_vec_b)
				

		output_vec_a = tf.concat(conv_vec_as, axis=-1)
		output_vec_b = tf.concat(conv_vec_bs, axis=-1)
		# [batch_size, 1, output_channel*3]

		output_a = tf.squeeze(output_vec_a, axis=-2)
		output_b = tf.squeeze(output_vec_b, axis=-2)
		# [batch_size, output_channel*3]
		
		# output_a = tf.reduce_mean(pool_vec_a, axis=-1)
		# output_b = tf.reduce_mean(pool_vec_b, axis=-1)

		mix_vec = tf.concat([output_a, output_b], axis=-1)

		#[batch_size, output_channel*3*4]
		drop_mix_vec = tf.nn.dropout(mix_vec, self.dropout)

		res = tf.layers.dense(drop_mix_vec, units=self.class_size, name='output-layer')

		return res

	def build_mid_layer_bilstm(self, input_vec_a, input_vec_b):

		fw_lstm_cell_a = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size, name="fw_lstm_cell_a")
		fw_lstm_cell_a = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell_a, output_keep_prob=0.5)
		bw_lstm_cell_a = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size, name="bw_lstm_cell_a")
		bw_lstm_cell_a = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell_a, output_keep_prob=0.5)
		fw_lstm_cell_b = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size, name="fw_lstm_cell_b")
		fw_lstm_cell_b = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell_b, output_keep_prob=0.5)
		bw_lstm_cell_b = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size, name="bw_lstm_cell_b")
		bw_lstm_cell_b = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell_b, output_keep_prob=0.5)

		bilstm_output_a, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_lstm_cell_a,
														cell_bw = bw_lstm_cell_a,
														inputs = input_vec_a,
														dtype = tf.float32)
		bilstm_output_b, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_lstm_cell_b,
														cell_bw = bw_lstm_cell_b,
														inputs = input_vec_b,
														dtype = tf.float32)

		output_a = tf.concat([bilstm_output_a[0][:,-1,:], bilstm_output_a[1][:,-1,:]], axis=-1)
		output_b = tf.concat([bilstm_output_b[0][:,-1,:], bilstm_output_b[1][:,-1,:]], axis=-1)
		# [batch_size, hidden_size*2]

		# output_a = tf.contrib.layers.layer_norm(output_a, begin_norm_axis=-1, scope="norm_output_a")
		# output_b = tf.contrib.layers.layer_norm(output_b, begin_norm_axis=-1, scope="norm_output_b")

		mix_vec = tf.concat([output_a, output_b], axis=-1)
		# [batch_size, hidden_size*8]


		mix_vec = tf.layers.dense(mix_vec, units=self.hidden_size, name='output-layer')
		drop_mix_vec = tf.nn.dropout(mix_vec, self.dropout)
		res = tf.layers.dense(drop_mix_vec, units=self.class_size, name='output-layer')

		return res
	
	def loss_op(self, input_vec):

		one_hot = tf.one_hot(self.label, depth=self.class_size)
		loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot,
														  logits=input_vec)
		return tf.reduce_mean(loss)