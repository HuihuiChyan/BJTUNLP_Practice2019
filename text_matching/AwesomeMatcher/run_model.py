import prepro
import model
import tensorflow as tf
import argparse
import os
import random
import pdb
import gc
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
# gpu_config = tf.ConfigProto(gpu_options=gpu_options)

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_path", type=str, default='data/vocab_all.txt')
parser.add_argument("--train_path", type=str, default='snli_1.0/snli_1.0_train.txt')
parser.add_argument("--dev_path", type=str, default='snli_1.0/snli_1.0_dev.txt')
parser.add_argument("--test_path", type=str, default='snli_1.0/snli_1.0_test.txt')
parser.add_argument("--model_path", type=str, default='model')

parser.add_argument("--maxlen", type=int, default=30)
parser.add_argument("--vocab_size", type=int, default=50000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--kernel", type=list, default=[3,4,5])
parser.add_argument("--class_size", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--hidden_size", type=int, default=256)

parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--steps_per_stat", type=int, default=10)
parser.add_argument("--steps_per_test", type=int, default=100)
parser.add_argument("--steps_per_save", type=int, default=100)

parser.add_argument("--is_train", type=int, default=1)


args = parser.parse_args()

def init_config(vocab_size):
	config = {}
	config['maxlen'] = args.maxlen
	config['hidden_size'] = args.hidden_size
	config['lr'] = args.lr
	config['vocab_size'] = args.vocab_size
	config['dropout'] = args.dropout
	config['kernel'] = args.kernel
	config['class_size'] = args.class_size
	config['batch_size'] = args.batch_size

	return config

def train():

	train_set = prepro.get_dataset(args.train_path)
	test_set = prepro.get_dataset(args.test_path)

	random.shuffle(train_set)

	if not os.path.exists(args.vocab_path):
		prepro.gen_vocab(train_set, args.vocab_path, args.vocab_size)

	idx2vocab = prepro.read_dict(args.vocab_path)
	vocab = {value:key for key,value in idx2vocab.items()}

	train_set_idx = prepro.turn2idx(train_set, vocab, maxlen=args.maxlen)
	test_set_idx = prepro.turn2idx(test_set, vocab, maxlen=args.maxlen)

	sess = tf.Session()

	config = init_config(vocab_size = len(vocab))
	infersent = model.InferSent(config)

	saver = tf.train.Saver(max_to_keep=5)

	if args.is_train == 1:
		init = tf.global_variables_initializer()
		print("Now create the session from beginning.")
		sess.run(init)
	else:
		ckpt = tf.train.get_checkpoint_state(args.model_path)
		saver.restore(sess, ckpt.model_checkpoint_path)

	sess.graph.finalize()
	
	step_count = 0

	best_test_acc = 0
	best_test_step = 0

	for i in range(args.epoch):
		start_train_time = time.time()
		for sent_a_batch, sent_b_batch, label_batch in prepro.generate_batch(train_set_idx[0],
			train_set_idx[1], train_set_idx[2], batch_size=args.batch_size, shuffle=True):
			feeding = {infersent.input1: sent_a_batch,
				   infersent.input2: sent_b_batch,
				   infersent.label: label_batch}
			acc_val, loss_val, _ = sess.run([infersent.acc, infersent.loss, infersent.train_op], feed_dict=feeding)
			step_count += 1
			if step_count%args.steps_per_stat == 0:
				end_train_time = time.time()
				train_time = end_train_time - start_train_time
				print("During step %d, the acc_val is %s, the loss_val is %s, time cost is %.2f secs." % (step_count, str(acc_val), str(loss_val), train_time))
				start_train_time = time.time()
			if step_count%args.steps_per_save == 0:
				print("Now saving model as checkpoint......")
				saver.save(sess, args.model_path+'/model.ckpt', global_step=step_count)
			if step_count%args.steps_per_test == 0:
				print("----Here is an evaluation on test set----")
				test_acc_val = []
				test_count = 0
				start_test_time = time.time()
				for test_a_batch, test_b_batch, test_label_batch in prepro.generate_batch(test_set_idx[0],
			test_set_idx[1], test_set_idx[2], batch_size=args.batch_size):
					feeding = {infersent.input1: test_a_batch,
						infersent.input2: test_b_batch,
						infersent.label: test_label_batch}
					acc_val = sess.run(infersent.acc, feed_dict=feeding)
					test_acc_val.append(acc_val)
					test_count += 1
				avg_test_acc = sum(test_acc_val) / len(test_acc_val)
				if avg_test_acc > best_test_acc:
					best_test_acc = avg_test_acc
					best_test_step = step_count
				# gc.collect()
				print("During step %d, the acc_val on test set is %s" % (step_count, str(avg_test_acc)))
				print("The best acc_val is %s, in step %d" % (best_test_acc, best_test_step))
				start_train_time = time.time()
	print("All training finished!")
	print("The best acc_val is %s, in step %d" % (best_test_acc, best_test_step))

train()