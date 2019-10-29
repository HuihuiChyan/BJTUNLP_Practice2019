import tensorflow as tf
from prepro import preprocess
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='2'

tf.flags.DEFINE_string("train_path","./data/train.txt","The whole path of the train file.")
tf.flags.DEFINE_string("test_path","./data/test.txt","The whole path of the test file.")
tf.flags.DEFINE_string("vocab_path","./data/vocab.txt","The whole path of the vocabulary file.")

tf.flags.DEFINE_integer("max_len",120,"The max length of the sequence.")
tf.flags.DEFINE_integer("batch_size",256,"The size of train batch.")
tf.flags.DEFINE_integer("vocab_size",4002,"The size of vocabulary.")

tf.flags.DEFINE_integer("embedding_size",200,"The embedding size.")
tf.flags.DEFINE_integer("rnn_size",256,"The size of lstm cell.")
tf.flags.DEFINE_float("output_keep_prob",0.7,"The keep probability of rnn output.")
tf.flags.DEFINE_integer("output_num",10,"The number of output classes.")
tf.flags.DEFINE_float("learning_rate",1e-3,"The learning_rate.")
tf.flags.DEFINE_float("norm_clip",5.0,"The clip norm.")

tf.flags.DEFINE_integer("max_train_step",50000,"The max train step number.")
tf.flags.DEFINE_integer("steps_per_stat",10,"Step number for every statement.")
tf.flags.DEFINE_integer("steps_per_eval",100,"Step number for every evaluation.")

FLAGS = tf.flags.FLAGS

class Tokenizer(object):

	char2idx = defaultdict(lambda :0) #0 for unk
	idx2char = {}
	tag2idx = {"B":0, "M":1, "E":2, "S":3}
	idx2tag = {0:"B", 1:"M", 2:"E", 3:"S", 4:"P"}

	def __init__(self, vocab_path):
		with open(vocab_path, "r", encoding="utf-8") as fvocab:
			vocablines = [line.strip() for line in fvocab.readlines()]
			for i,line in enumerate(vocablines):
				self.char2idx[line] = i
				self.idx2char[i] = line

	def tokenize(self, lines):
		lines = [line.split(" ||| ") for line in lines]

		newlines = []
		for i,line in enumerate(lines):
			textline = list(line[0])
			tagline = list(line[1])

			assert len(textline) == len(tagline)

			textline = [self.char2idx[char] for char in textline]
			tagline = [self.tag2idx[tag] for tag in tagline]
			if len(textline) > FLAGS.max_len:
				textline = textline[:FLAGS.max_len]
				tagline = tagline[:FLAGS.max_len]
				this_len = FLAGS.max_len
			elif len(textline) < FLAGS.max_len:
				this_len = len(textline)
				textline.extend([1 for _ in range(FLAGS.max_len - len(textline))])
				tagline.extend([4 for _ in range(FLAGS.max_len - len(tagline))])				
			else:
				this_len = FLAGS.max_len

			newlines.append((textline, tagline, this_len))

		return newlines

def iterator_creator(trainlines, is_training=True):

	text_lines = [line[0] for line in trainlines]
	tag_lines = [line[1] for line in trainlines]
	len_lines = [line[2] for line in trainlines]	
	text_dataset = tf.data.Dataset.from_tensor_slices(text_lines)
	tag_dataset = tf.data.Dataset.from_tensor_slices(tag_lines)
	len_dataset = tf.data.Dataset.from_tensor_slices(len_lines)

	dataset = tf.data.Dataset.zip((text_dataset, tag_dataset, len_dataset))

	if is_training==True:
		dataset = dataset.repeat()

	dataset = dataset.batch(FLAGS.batch_size)

	iterator = dataset.make_initializable_iterator()

	return iterator


class AwesomeSegmenter(object):

	def __init__(self, iterator, reuse):

		next_batch = iterator.get_next()

		self.reuse = reuse

		self.batched_text = next_batch[0]
		self.batched_tag = next_batch[1]
		self.batched_len = next_batch[2]
		self.build_graph()

	def build_graph(self):
		with tf.variable_scope("embedding_layer", reuse=self.reuse):

			embedding_table = tf.get_variable("embedding_table",
											  initializer=tf.contrib.layers.xavier_initializer(),
											  dtype=tf.float32,
											  shape=[FLAGS.vocab_size, FLAGS.embedding_size])

			embedded_input = tf.nn.embedding_lookup(embedding_table, self.batched_text)

		with tf.variable_scope("bilstm_layer", reuse=self.reuse):

			fw_lstmcell = tf.nn.rnn_cell.BasicLSTMCell(num_units=FLAGS.rnn_size, reuse=self.reuse)
			fw_lstmcell = tf.nn.rnn_cell.DropoutWrapper(fw_lstmcell, output_keep_prob=0.7)
			bw_lstmcell = tf.nn.rnn_cell.BasicLSTMCell(num_units=FLAGS.rnn_size, reuse=self.reuse)
			bw_lstmcell = tf.nn.rnn_cell.DropoutWrapper(bw_lstmcell, output_keep_prob=0.7)

			output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstmcell,
													 cell_bw=bw_lstmcell,
													 inputs=embedded_input,
													 dtype=tf.float32)
			concat_output = tf.concat(output, axis=-1)

		with tf.variable_scope("final_layer", reuse=self.reuse) as scope:

			self.final_output = tf.layers.dense(concat_output, FLAGS.output_num, reuse=self.reuse)
			self.output_logits = tf.argmax(self.final_output, axis=-1)
			self.loss = self.loss_op(scope)
			self.train_step = self.train_op(scope)
			self.metric = self.metric_op(scope)


	def train_op(self, scope):

		optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
		grads_and_vars = optimizer.compute_gradients(self.loss)

		for i,(g,v) in enumerate(grads_and_vars):
			if g is not None:
				grads_and_vars[i] = (tf.clip_by_norm(g, FLAGS.norm_clip), v)

		train_op = optimizer.apply_gradients(grads_and_vars)
		return train_op


	def loss_op(self, scope):

		seq_mask = tf.sequence_mask(self.batched_len, maxlen=FLAGS.max_len)
		losses = tf.losses.softmax_cross_entropy(tf.one_hot(self.batched_tag, depth=FLAGS.output_num), self.final_output, weights=seq_mask)
		loss = tf.reduce_mean(losses)
		return loss

	def metric_op(self, scope):

		seq_mask = tf.sequence_mask(self.batched_len, maxlen=FLAGS.max_len)
		acc = tf.metrics.accuracy(labels=self.batched_tag, predictions=self.output_logits, weights=seq_mask)
		return acc


def main():
	with open(FLAGS.train_path, "r", encoding="utf-8") as ftrain,\
	open(FLAGS.test_path, "r", encoding="utf-8") as ftest:
		trainlines = [line.strip() for line in ftrain.readlines()]
		testlines = [line.strip() for line in ftest.readlines()]

	tokenizer = Tokenizer(FLAGS.vocab_path)
	trainlines = tokenizer.tokenize(trainlines)
	testlines = tokenizer.tokenize(testlines)

	train_iterator = iterator_creator(trainlines)
	test_iterator = iterator_creator(testlines, is_training=False)
	
	train_model = AwesomeSegmenter(train_iterator, reuse=False)
	test_model = AwesomeSegmenter(test_iterator, reuse=True)

	best_precision = 0.
	best_recall = 0.
	best_f1 = 0.
	best_step = 0

	epoch_count = 0

	with tf.Session() as sess:
		sess.run(tf.initializers.global_variables())
		sess.run(tf.local_variables_initializer())
		sess.run(train_iterator.initializer)
		sess.run(tf.tables_initializer())
		for i in range(FLAGS.max_train_step):
			try:
				_, loss_value, acc_value = sess.run([train_model.train_step, train_model.loss, train_model.metric])
			except:
				epoch_count += 1
				print("One epoch finished. Now the following is %d epoch." % (epoch_count))
				sess.run(train_iterator.initializer)
			if i % FLAGS.steps_per_stat == 0:
				print("At step %d, loss is %.4f, accuracy is %.4f." % (i, loss_value, acc_value[0]))
			if i % FLAGS.steps_per_eval == 0:
				print("The following is an evaluation on test set:")
				sess.run(test_iterator.initializer)
				total_logits = []
				total_tags = []
				while(True):
					try:
						output_logits, batched_tag, batched_len = sess.run([test_model.output_logits, test_model.batched_tag, test_model.batched_len])
						for j in range(len(batched_len)):
							this_len = batched_len[j]
							total_logits.extend(output_logits[j][:this_len].tolist())
							total_tags.extend(batched_tag[j][:this_len].tolist())
					except:
						f1_value = f1_score(y_true=total_tags, y_pred=total_logits, labels=[0,1,2,3], average='micro')
						precision_value = precision_score(y_true=total_tags, y_pred=total_logits, labels=[0,1,2,3], average='micro')
						recall_value = recall_score(y_true=total_tags, y_pred=total_logits, labels=[0,1,2,3], average='micro')
						print("At step %d, the precision is %.4f, recall is %.4f, f1 score is: %.4f." % (i, precision_value, recall_value, f1_value))
						if f1_value > best_f1:
							best_precision = precision_value
							best_recall = recall_value
							best_f1 = f1_value
							best_step = i
						print("The best step is %d, the precision is %.4f, recall is %.4f, f1 score is: %.4f." % (best_step, best_precision, best_recall, best_f1))
						break
		print("All training finished! Totally %d epochs." % (epoch_count))
		print("The best step is %d, the precision is %.4f, recall is %.4f, f1 score is: %.4f." % (best_step, best_precision, best_recall, best_f1))

if __name__ == "__main__":
	main()