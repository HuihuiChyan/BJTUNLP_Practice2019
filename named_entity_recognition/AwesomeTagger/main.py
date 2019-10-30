import os
import pdb
import re
import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tf.flags.DEFINE_string("train_src", "data/eng.train.src.shuf", "The shuffled source training file.")
tf.flags.DEFINE_string("train_tgt", "data/eng.train.trg.shuf", "The shuffled label training file.")
tf.flags.DEFINE_string("dev_src", "data/eng.testa.src", "The shuffled source develop file.")
tf.flags.DEFINE_string("dev_tgt", "data/eng.testa.trg", "The shuffled label develop file.")
tf.flags.DEFINE_string("test_src", "data/eng.testb.src", "The shuffled source test file.")
tf.flags.DEFINE_string("test_tgt", "data/eng.testb.trg", "The shuffled target test file.")
tf.flags.DEFINE_string("vocab_src", "data/vocab.w", "The vocabulary for words in source file.")
tf.flags.DEFINE_string("vocab_tgt", "data/vocab.t", "The vocabulary for labels in target file.")
tf.flags.DEFINE_string("predict_path", "data/predicted.txt", "The predicted file used for conlleval.")
tf.flags.DEFINE_string("metric_path", "data/metrics.txt", "The metric_path for conlleval.")
tf.flags.DEFINE_string("perl_path", "conlleval_rev.pl", "The path for perl file conlleval.pl.")

tf.flags.DEFINE_integer("max_len", 50, "The max length of per line.")
tf.flags.DEFINE_integer("batch_size", 64, "The size of each batch.")
tf.flags.DEFINE_integer("embedding_size", 256, "The embedding size.")
tf.flags.DEFINE_integer("lstm_hidden_size", 256, "The hidden size of LSTM cell.")
tf.flags.DEFINE_float("lstm_keep_prob", 0.7, "The keep probability of LSTM cell.")
tf.flags.DEFINE_float("learning_rate", 0.001, "The learning rate of AdamOptimizer.")
tf.flags.DEFINE_integer("norm_clip", 5, "The upper bound of gradients.")

tf.flags.DEFINE_integer("steps_per_dev", 100, "The step number for each evaluation on dev set.")
tf.flags.DEFINE_integer("steps_per_test", 100, "The step number for each evaluation on test set.")
tf.flags.DEFINE_integer("steps_per_stat", 10, "The step number for each statement.")
tf.flags.DEFINE_integer("max_train_steps", 100000, "The max train step number.")

FLAGS = tf.flags.FLAGS

LABEL_NUM = 18

BEST_PRECISION = 0.
BEST_RECALL = 0.
BEST_F1 = 0.
BEST_STEP = 0

class WordTokenizer:

	word2idx = defaultdict(lambda :2) #2 for unk
	idx2word = {}

	def __init__(self, vocab):
		for i,word in enumerate(vocab):
			self.word2idx[word] = i
			self.idx2word[i] = word
	def tokenize(self, line):
		idx_line = []
		for word in line.split():
			idx_line.append(self.word2idx[word])
		line_length = len(idx_line)
		if len(idx_line) > FLAGS.max_len:
			idx_line = idx_line[:FLAGS.max_len]
		elif len(idx_line) < FLAGS.max_len:
			idx_line = idx_line + [0 for _ in range(FLAGS.max_len-len(idx_line))] #0 for pad
		return idx_line, line_length
	def retokenize(self, line):
		word_line = []
		for idx in line.split():
			word_line.append(self.idx2word[idx])
		return word_line

class LabelTokenizer:

	word2idx = {}
	idx2word = {}

	def __init__(self, vocab):
		for i,word in enumerate(vocab):
			self.word2idx[word] = i
			self.idx2word[i] = word
	def tokenize(self, line):
		idx_line = []
		for word in line.split():
			idx_line.append(self.word2idx[word])
		if len(idx_line) > FLAGS.max_len:
			idx_line = idx_line[:FLAGS.max_len]
		elif len(idx_line) < FLAGS.max_len:
			idx_line = idx_line + [0 for _ in range(FLAGS.max_len-len(idx_line))] #0 for pad(actually <eos> in label lines)
		return idx_line
	def retokenize(self, line):
		word_line = []
		for idx in line:
			if idx == 0:
				word_line.append("O")
			else:
				word_line.append(self.idx2word[idx])
		return word_line

def iterator_creator(src_lines, tgt_lines, wordTokenizer, labelTokenizer, is_training=True):

	src_len_lines = [wordTokenizer.tokenize(line) for line in src_lines]
	tgt_lines = [labelTokenizer.tokenize(line) for line in tgt_lines]
	src_lines = [line[0] for line in src_len_lines]
	len_lines = [line[1] for line in src_len_lines]
	src_dataset = tf.data.Dataset.from_tensor_slices(src_lines)
	tgt_dataset = tf.data.Dataset.from_tensor_slices(tgt_lines)
	len_dataset = tf.data.Dataset.from_tensor_slices(len_lines)
	dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, len_dataset))
	dataset = dataset.batch(FLAGS.batch_size)
	if is_training:
		dataset = dataset.repeat()
	iterator = dataset.make_initializable_iterator()

	return iterator

class AwesomeTagger:

	def __init__(self, src_vocab_size, tgt_vocab_size, iterator, reuse):
		next_batch = iterator.get_next()
		batched_src = next_batch[0]
		batched_tgt = next_batch[1]
		batched_len = next_batch[2]
		self.batched_src = next_batch[0]
		self.batched_tgt = next_batch[1]
		self.batched_len = next_batch[2]
		with tf.variable_scope("embedding_layer", reuse=reuse):
			embedding = tf.get_variable("embedding", initializer=tf.contrib.layers.xavier_initializer(),\
										shape=[src_vocab_size, FLAGS.embedding_size], dtype=tf.float32)
			embedded_seq = tf.nn.embedding_lookup(embedding, batched_src)
		with tf.variable_scope("bilstm_layer", reuse=reuse):
			fwLstmCell = tf.nn.rnn_cell.BasicLSTMCell(num_units=FLAGS.lstm_hidden_size, forget_bias=0.7, reuse=reuse)
			fwLstmCell = tf.nn.rnn_cell.DropoutWrapper(fwLstmCell, output_keep_prob=FLAGS.lstm_keep_prob)
			bwLstmCell = tf.nn.rnn_cell.BasicLSTMCell(num_units=FLAGS.lstm_hidden_size, forget_bias=0.7, reuse=reuse)
			bwLstmCell = tf.nn.rnn_cell.DropoutWrapper(bwLstmCell, output_keep_prob=FLAGS.lstm_keep_prob)
			self.outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwLstmCell,\
															 cell_bw=bwLstmCell,\
															 inputs=embedded_seq,\
															 #sequence_length=batched_len,\
															 dtype=tf.float32)
			concat_outputs = tf.concat([self.outputs[0], self.outputs[1]], -1)
		with tf.variable_scope("final_layer",reuse=reuse):
			self.global_step = tf.get_variable("global_step", initializer=0, trainable=False, dtype=tf.int32)
			self.sequence_mask = tf.sequence_mask(batched_len, maxlen=FLAGS.max_len)
			self.final_outputs = tf.layers.dense(concat_outputs, units=tgt_vocab_size, reuse=reuse)
			self.final_logits = tf.arg_max(self.final_outputs, dimension=-1)
			self.labels = tf.one_hot(batched_tgt, depth=tgt_vocab_size)
			#self.labels = tf.boolean_mask(self.labels, sequence_mask)
			#self.final_outputs = tf.boolean_mask(self.final_outputs, sequence_mask)
			self.losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.final_outputs)
			self.masked_losses = tf.boolean_mask(self.losses, self.sequence_mask)
			self.loss = tf.reduce_mean(self.masked_losses)
			#self.loss = tf.reduce_mean(self.losses)
			optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			grads_and_vars = optimizer.compute_gradients(self.loss)
			for i, (g, v) in enumerate(grads_and_vars):
				if g is not None:
					grads_and_vars[i] = (tf.clip_by_norm(g, FLAGS.norm_clip), v)
			self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

# def marco_f1(batched_labels, batched_logits):

# 	masked_labels = 1 - np.equal(batched_labels, 0)
# 	f1_scores = []
# 	epsilon = 1e-7
# 	for i in range(LABEL_NUM)[1:]:
# 		real_labels = np.equal(i, batched_labels)
# 		real_logits = np.equal(i, batched_logits)
# 		true_pos = np.sum(np.logical_and(np.logical_and(real_labels, real_logits), masked_labels))
# 		false_pos = np.sum(np.logical_and(np.logical_and(real_labels, 1-real_logits), masked_labels))
# 		false_neg = np.sum(np.logical_and(np.logical_and(1-real_labels, real_logits), masked_labels))
# 		precision = true_pos / (true_pos + false_pos + epsilon)
# 		recall = true_pos / (true_pos + false_neg + epsilon)
# 		f1_score = 2 * precision * recall / (precision + recall + epsilon)
# 		f1_scores.append(f1_score)

# 	marco_f1_score = np.mean(f1_scores)
# 	# pdb.set_trace()

# 	return marco_f1_score


def BIOES_2_BIO(tag):
	if tag[0] == 'E':
		return 'I'+tag[1:]
	elif tag[0] == 'S':
		return 'B'+tag[1:]
	elif tag == 'O':
		return '0'
	else:
		return tag

def conlleval(srclines, tgtlines, all_logits, labelTokenizer, global_step, is_test=False):

	global BEST_PRECISION, BEST_RECALL, BEST_F1, BEST_STEP

	with open(FLAGS.predict_path, "w", encoding="utf-8") as fpre:
		srclines = [line.split() for line in srclines]
		tgtlines = [line.split() for line in tgtlines]
		all_logits = [labelTokenizer.retokenize(line) for line in all_logits]
		for i in range(len(all_logits)):
			assert len(srclines[i]) == len(tgtlines[i])
			assert len(all_logits[i]) == FLAGS.max_len
			loop = len(srclines[i]) if len(srclines[i])<=50 else 50
			for j in range(loop):
				fpre.write(str(srclines[i][j])+" "+BIOES_2_BIO(str(all_logits[i][j]))+" "+BIOES_2_BIO(str(tgtlines[i][j]))+"\n")
	os.system("perl {} < {} > {}".format(FLAGS.perl_path, FLAGS.predict_path, FLAGS.metric_path))

	with open(FLAGS.metric_path, "r", encoding="utf-8") as fme:
		melines = [line.strip() for line in fme.readlines()]
		for line in melines:
			print(line)
			if is_test == True:
				if re.match("accuracy:  [0-9]+\.[0-9]+%; precision:  [0-9]+\.[0-9]+%; recall:  [0-9]+\.[0-9]+%; FB1:  [0-9]+\.[0-9]+", line):
					result = re.findall("[0-9]+\.[0-9]+",line)
					try:
						assert len(result) == 4
					except:
						pdb.set_trace()
					if BEST_F1 < float(result[3]):
						BEST_PRECISION = float(result[1])
						BEST_RECALL = float(result[2])
						BEST_F1 = float(result[3])
						BEST_STEP = global_step
		# global BEST_PRECISION, BEST_RECALL, BEST_F1, BEST_STEP	
		# print("The best result on test set till now is:")
		# print("At step %d, best precision: %f, best recall: %f, best f1 value: %f." % (BEST_STEP, BEST_PRECISION, BEST_RECALL, BEST_F1))


def main(_):
	with open(FLAGS.vocab_src, "r", encoding="utf-8") as fsrcvocab,\
	open(FLAGS.vocab_tgt, "r", encoding="utf-8") as ftgtvocab:
		src_vocab = [line.strip() for line in fsrcvocab.readlines()]
		tgt_vocab = [line.strip() for line in ftgtvocab.readlines()]

	src_vocab_size = len(src_vocab)
	tgt_vocab_size = len(tgt_vocab)

	wordTokenizer = WordTokenizer(src_vocab)
	labelTokenizer = LabelTokenizer(tgt_vocab)

	with open(FLAGS.train_src, "r", encoding="utf-8") as fsrctrain,\
	open(FLAGS.train_tgt, "r", encoding="utf-8") as ftgttrain,\
	open(FLAGS.dev_src, "r", encoding="utf-8") as fsrcdev,\
	open(FLAGS.dev_tgt, "r", encoding="utf-8") as ftgtdev,\
	open(FLAGS.test_src, "r", encoding="utf-8") as fsrctest,\
	open(FLAGS.test_tgt, "r", encoding="utf-8") as ftgttest:

		srctrainlines = [line.strip() for line in fsrctrain.readlines()]
		tgttrainlines = [line.strip() for line in ftgttrain.readlines()]
		srcdevlines = [line.strip() for line in fsrcdev.readlines()]
		tgtdevlines = [line.strip() for line in ftgtdev.readlines()]
		srctestlines = [line.strip() for line in fsrctest.readlines()]
		tgttestlines = [line.strip() for line in ftgttest.readlines()]

	train_iterator = iterator_creator(srctrainlines, tgttrainlines, wordTokenizer, labelTokenizer)
	dev_iterator = iterator_creator(srcdevlines, tgtdevlines, wordTokenizer, labelTokenizer, is_training=False)
	test_iterator = iterator_creator(srctestlines, tgttestlines, wordTokenizer, labelTokenizer, is_training=False)

	train_model = AwesomeTagger(src_vocab_size, tgt_vocab_size, train_iterator, reuse=False)
	dev_model = AwesomeTagger(src_vocab_size, tgt_vocab_size, dev_iterator, reuse=True)
	test_model = AwesomeTagger(src_vocab_size, tgt_vocab_size, test_iterator, reuse=True)

	with tf.Session() as sess:
		sess.run(train_iterator.initializer)	
		sess.run(tf.tables_initializer())
		sess.run(tf.initializers.global_variables())
		sess.run(tf.initializers.local_variables())
		best_f1_score = 0.0
		best_global_step = 0
		while(1):
			global_step = sess.run(train_model.global_step)
			if global_step == FLAGS.max_train_steps:
				break
			if global_step % FLAGS.steps_per_dev == 0:
				print("Now an evaluation on dev set starts:")
				sess.run(dev_iterator.initializer)
				all_logits = []
				while(1):
					try:
						this_logits = sess.run([dev_model.final_logits])
						all_logits.extend(this_logits[0].tolist())
					except:
						#pdb.set_trace()
						break
				print("The following is an evaluation on dev set:")
				conlleval(srcdevlines, tgtdevlines, all_logits, labelTokenizer, global_step)
				#print("At global step %d, the f1 score on dev set is %s." % (global_step, str(f1_score)))
			if global_step % FLAGS.steps_per_test == 0:
				print("Now an evaluation on test set starts:")
				sess.run(test_iterator.initializer)
				all_logits = []
				while(1):
					try:
						this_logits = sess.run([test_model.final_logits])
						all_logits.extend(this_logits[0].tolist())
					except:
						#pdb.set_trace()
						break
				print("The following is an evaluation on test set:")
				conlleval(srctestlines, tgttestlines, all_logits, labelTokenizer, global_step, is_test=True)
				#print("At global step %d, the f1 score on test set is %s." % (global_step, str(f1_score)))
				#print("The best f1 score is %s, at global step %d." % (str(best_f1_score), best_global_step))
				#if f1_score > best_f1_score:
					#best_global_step = global_step
					#best_f1_score = f1_score
				global BEST_PRECISION, BEST_RECALL, BEST_F1, BEST_STEP	
				print("The best result on test set till now is:")
				print("At step %d, best precision: %.2f, best recall: %.2f, best f1 value: %.2f." % (BEST_STEP, BEST_PRECISION, BEST_RECALL, BEST_F1))

			_, loss_value = sess.run([train_model.train_op, train_model.loss])
			if global_step%FLAGS.steps_per_stat == 0:
				print("At global step %d, the loss is %s" % (global_step, str(loss_value)))

		print("All training finished.")
		print("At step %d, best precision: %.2f, best recall: %.2f, best f1 value: %.2f." % (BEST_STEP, float(result[1]), float(result[2]), float(result[3])))

if __name__ == "__main__":
	tf.app.run()
