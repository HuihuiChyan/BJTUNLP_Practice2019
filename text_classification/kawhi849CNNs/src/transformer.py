import tensorflow as tf
import numpy as np
import math
from read import generator


class Transformer(object):
	def __init__(self, vocab_size, seq_len):
		self.batch_size = 32
		self.seq_len = seq_len
		self.classes = 2
		self.embedding_size = 128
		self.vocab_size = vocab_size
		self.placeholder()
		self.embedding()
		self.transformer()
		self.fc_layer()
		self.compute_loss()

	def placeholder(self):
		self.inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
		self.labels = tf.placeholder(shape=[None, None], dtype=tf.int32)
		self.dropout_ = tf.placeholder(dtype=tf.float32)
				
	def embedding(self):
		embeddings = tf.get_variable(name='word_emb', shape=[self.vocab_size, self.embedding_size],
					#[self.vocab_size, self.embedding_size],
					#tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
					initializer=tf.truncated_normal_initializer(stddev=1),dtype=tf.float32)
		_word_embeddings = tf.Variable(embeddings, dtype=tf.float32, trainable=True, name="embeddings_table")
		self.word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.inputs, name="word_embeddings")
		# self.embedded_chars_expanded = tf.expand_dims(word_embeddings, -1)

	def transformer(self):
		num_attention_heads = 4
		num_hidden_layers = 3
		initializer_range = 0.02
		hidden_dropout_prob = self.dropout_
		do_return_all_layers = True
		intermediate_act_fn = gelu
		input_tensor = self.word_embeddings
		attention_head_size = int(self.embedding_size / num_attention_heads)
		input_shape = self.get_shape_list(input_tensor, expected_rank=3)
		batch_size = self.batch_size
		seq_length = self.seq_len
		input_width = self.embedding_size
		
		prev_output = self.reshape_to_matrix(input_tensor)

		all_layer_outputs = []
		for layer_idx in range(num_hidden_layers):
			with tf.variable_scope("layer_%d" % layer_idx):
				layer_input = prev_output

				with tf.variable_scope("attention"):
					attention_heads = []
					with tf.variable_scope("self"):
						attention_head = self.attention_layer(
									from_tensor=layer_input,
									to_tensor=layer_input,
									num_attention_heads=num_attention_heads,
									size_per_head=attention_head_size,
									attention_probs_dropout_prob=hidden_dropout_prob,
									initializer_range=initializer_range,
									do_return_2d_tensor=True,
									batch_size=batch_size,
									from_seq_length=seq_length,
									to_seq_length=seq_length)
						attention_heads.append(attention_head)

					attention_output = None
					if len(attention_heads) == 1:
						attention_output = attention_heads[0]
					else:
						attention_output = tf.concat(attention_heads, axis=-1)

					with tf.variable_scope("output"):
						attention_output = tf.layers.dense(
									attention_output,
									self.embedding_size,
									name='attention_output',
									kernel_initializer=self.create_initializer(initializer_range))
						attention_output = self.dropout(attention_output, hidden_dropout_prob)
						attention_output = self.layer_norm(attention_output + layer_input)
			
				with tf.variable_scope("intermediate"):
					intermediate_output = tf.layers.dense(
							attention_output,
							self.embedding_size * 4,
							activation=intermediate_act_fn,
							name='inter_output',
							kernel_initializer=self.create_initializer(initializer_range))
				with tf.variable_scope("output"):
					layer_output = tf.layers.dense(
							intermediate_output,
							self.embedding_size,
							name='layer_output',
							kernel_initializer=self.create_initializer(initializer_range))
					layer_output = self.dropout(layer_output, hidden_dropout_prob)
					layer_output = self.layer_norm(layer_output + attention_output)
					prev_output = layer_output
					all_layer_outputs.append(layer_output)
		self.feature = prev_output
		print(self.feature.shape)
		if do_return_all_layers:
			final_outputs = []
			for layer_output in all_layer_outputs:
				final_output = self.reshape_from_matrix(layer_output, input_shape)
				final_outputs.append(final_output)
			return final_outputs
		else:
			final_output = self.reshape_from_matrix(prev_output, input_shape)
			return final_output

	def attention_layer(self,
				from_tensor,
				to_tensor,
				num_attention_heads=1,
				size_per_head=512,
				query_act=None,
				key_act=None,
				value_act=None,
				attention_probs_dropout_prob=0,
				initializer_range=0.02,
				do_return_2d_tensor=False,
				batch_size=None,
				from_seq_length=None,
				to_seq_length=None):

		
		def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
			output_tensor = tf.reshape(
					input_tensor, [-1, seq_length, num_attention_heads, width])
			output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
			return output_tensor

		from_shape = self.get_shape_list(from_tensor, expected_rank=[2, 3])
		to_shape = self.get_shape_list(to_tensor, expected_rank=[2, 3])


		if len(from_shape) == 3:
			batch_size = from_shape[0]
			from_seq_length = from_shape[1]
			to_seq_length = to_shape[1]
		elif len(from_shape) == 2:
			if (batch_size is None or from_seq_length is None or to_seq_length is None):
				raise ValueError(
						"When passing in rank 2 tensors to attention_layer, the values "
						"for `batch_size`, `from_seq_length`, and `to_seq_length` "
						"must all be specified.")

		# Scalar dimensions referenced here:
		#	 B = batch size (number of sequences)
		#	 F = `from_tensor` sequence length
		#	 T = `to_tensor` sequence length
		#	 N = `num_attention_heads`
		#	 H = `size_per_head`

		from_tensor_2d = self.reshape_to_matrix(from_tensor)
		to_tensor_2d = self.reshape_to_matrix(to_tensor)

		# `query_layer` = [B*F, N*H]
		query_layer = tf.layers.dense(
				from_tensor_2d,
				num_attention_heads * size_per_head,
				activation=query_act,
				name="query",
				kernel_initializer=self.create_initializer(initializer_range))

		# `key_layer` = [B*T, N*H]
		key_layer = tf.layers.dense(
				to_tensor_2d,
				num_attention_heads * size_per_head,
				activation=key_act,
				name="key",
				kernel_initializer=self.create_initializer(initializer_range))

		# `value_layer` = [B*T, N*H]
		value_layer = tf.layers.dense(
				to_tensor_2d,
				num_attention_heads * size_per_head,
				activation=value_act,
				name="value",
				kernel_initializer=self.create_initializer(initializer_range))

		# `query_layer` = [B, N, F, H]
		query_layer = transpose_for_scores(query_layer, batch_size, num_attention_heads, from_seq_length, size_per_head)

		# `key_layer` = [B, N, T, H]
		key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads, to_seq_length, size_per_head)

		# Take the dot product between "query" and "key" to get the raw
		# attention scores.
		# `attention_scores` = [B, N, F, T]
		attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
		attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

		# Normalize the attention scores to probabilities.
		# `attention_probs` = [B, N, F, T]
		attention_probs = tf.nn.softmax(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs, attention_probs_dropout_prob)

		# `value_layer` = [B, T, N, H]
		value_layer = tf.reshape(
				value_layer,
				[-1, to_seq_length, num_attention_heads, size_per_head])

		# `value_layer` = [B, N, T, H]
		value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

		# `context_layer` = [B, N, F, H]
		context_layer = tf.matmul(attention_probs, value_layer)

		# `context_layer` = [B, F, N, H]
		context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

		if do_return_2d_tensor:
			# `context_layer` = [B*F, N*H]
			context_layer = tf.reshape(
					context_layer,
					[-1, num_attention_heads * size_per_head])
		else:
			# `context_layer` = [B, F, N*H]
			context_layer = tf.reshape(
					context_layer,
					[-1, from_seq_length, num_attention_heads * size_per_head])

		return context_layer

	def reshape_to_matrix(self, input_tensor):
		"""Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
		ndims = input_tensor.shape.ndims
		if ndims < 2:
			raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
											 (input_tensor.shape))
		if ndims == 2:
			return input_tensor

		width = input_tensor.shape[-1]
		output_tensor = tf.reshape(input_tensor, [-1, width])
		return output_tensor


	def reshape_from_matrix(self, output_tensor, orig_shape_list):
		"""Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
		if len(orig_shape_list) == 2:
			return output_tensor

		output_shape = self.get_shape_list(output_tensor)

		orig_dims = orig_shape_list[0:-1]
		width = output_shape[-1]

		return tf.reshape(output_tensor, orig_dims + [width])


	def dropout(self, input_tensor, dropout_prob):
		if dropout_prob is None or dropout_prob == 0.0:
			return input_tensor
		#output = input_tensor
		output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
		return output


	def layer_norm(self, input_tensor, name=None):
		return tf.contrib.layers.layer_norm(
				inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


	def layer_norm_and_dropout(self, input_tensor, dropout_prob, name=None):
		output_tensor = self.layer_norm(input_tensor, name)
		output_tensor = self.dropout(output_tensor, dropout_prob)
		return output_tensor
	

	def create_initializer(self, initializer_range=0.02):
		return tf.truncated_normal_initializer(stddev=initializer_range)

	def get_shape_list(self, tensor, expected_rank=None, name=None):
		if name is None:
			name = tensor.name

		#if expected_rank is not None:
		#	assert_rank(tensor, expected_rank, name)

		shape = tensor.shape.as_list()
		non_static_indexes = []
		for (index, dim) in enumerate(shape):
			if dim is None:
				non_static_indexes.append(index)
		if not non_static_indexes:
			return shape

		dyn_shape = tf.shape(tensor)
		for index in non_static_indexes:
			shape[index] = dyn_shape[index]
		return shape



	def fc_layer(self):
		w = tf.get_variable(name='w',shape=[self.seq_len, self.classes], initializer=tf.truncated_normal_initializer(stddev=0.02), dtype=tf.float32)
		b = tf.get_variable(name='b', initializer=tf.random_uniform([self.classes], -0.1, 0.1), dtype=tf.float32)
		output = tf.reshape(
			tf.layers.dense(self.feature, 1, activation=tf.tanh, kernel_initializer=self.create_initializer()), [-1, self.seq_len])
		self.logits = tf.matmul(output,w) + b

	def compute_loss(self):
		global_step = tf.Variable(0, trainable=False)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
		self.train_op = self.optimizer.minimize(self.loss)
		self.pred = tf.argmax(self.logits, 1)

	# def attention(self):
	def train(self, data, labels, test_data, test_labels, sess, epoches):
		best_val = 0.0
		for e in range(epoches):
			Loss = 0
			results = []
			batches = generator(zip(data, labels), self.batch_size)
			for step, (sent, label) in enumerate(batches):
				loss, _, res = sess.run([self.loss, self.train_op, self.pred],															
								feed_dict={self.inputs:np.array(sent), self.labels:np.array(label), self.dropout_:0})
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
			res = sess.run(self.pred, feed_dict={self.inputs:np.array(sent), self.labels:np.array(label), self.dropout_:0})
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

def gelu(x):
	cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
	return x * cdf





