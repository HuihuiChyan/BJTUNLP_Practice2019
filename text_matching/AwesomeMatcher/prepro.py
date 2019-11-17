#encoding=utf-8
import numpy as np
from collections import Counter
import pdb

def generate_batch(sent_a_idxs, sent_b_idxs, label, batch_size=4, shuffle=False):
	length = len(label)
	if shuffle:
		permutation = np.random.permutation(length)
		sent_a_idxs = sent_a_idxs[permutation]
		sent_b_idxs = sent_b_idxs[permutation]
		label = label[permutation]
	size = (length-1)//batch_size+1
	for i in range(size):
		begin = i*batch_size
		end = min((i+1)*batch_size, length)
		yield sent_a_idxs[begin:end], sent_b_idxs[begin:end], label[begin:end]

def sent2idx(sent, vocab, maxlen=10):
	sent2idxs = np.zeros(maxlen, dtype=np.int32)
	for idx, word in enumerate(sent.split()):
		sent2idxs[idx] = vocab.get(word.lower(), 1)
		if idx == maxlen-1:
			break
	return sent2idxs

def turn2idx(data_set, vocab, maxlen):
	sent_a_idxs = []
	sent_b_idxs = []
	label = []
	for item in data_set:
		text_a_idx = sent2idx(item.text_a, vocab, maxlen)
		text_b_idx = sent2idx(item.text_b, vocab, maxlen)
		sent_a_idxs.append(text_a_idx)
		sent_b_idxs.append(text_b_idx)
		label.append(item.label)
	Toarray = lambda x:np.array(x, dtype=np.int32)
	return Toarray(sent_a_idxs), Toarray(sent_b_idxs), Toarray(label)

def read_dict(path):
	vocab = dict()
	with open(path, "r", encoding="utf-8") as fr:
		data = fr.read().split("\n")[:-1]
	for idx, word in enumerate(data):
		vocab[idx+1] = word
	vocab[0] = '[PAD]'
	return vocab

class InputExample(object):

	def __init__(self, text_a, text_b, label, idx):
		self.text_a = text_a
		self.text_b = text_b
		self.idx = idx
		if label == 'contradiction':
			self.label = 0
		elif label == 'neutral':
			self.label = 1
		elif label == 'entailment':
			self.label = 2
		else:
			self.label = 3 #There are a few lines with '-' as their label, and I just delete them.

def get_dataset(path):
	data = []
	res = []
	with open(path, 'r', encoding="utf-8") as fr:
		data = fr.read().split('\n')[1:-1]

	for idx,line in enumerate(data):
		line = line.strip().split('\t')
		example = InputExample(line[5], line[6], line[0], idx)
		if example.label != 3:
			res.append(example)
	return res

def gen_vocab(data_set, vocab_path, vocab_size):
	all_sents = []
	vocab_words = ['UNK']
	for item in data_set:
		all_sents.extend(item.text_a.split())
		all_sents.extend(item.text_b.split())
	all_sents = [word.lower() for word in all_sents]
	counter = Counter(all_sents)
	common_words = counter.most_common()
	print("Totally %d words and we pick out %d words." % (len(common_words), vocab_size))
	vocab_words.extend([word[0] for word in common_words[:vocab_size-2]])
	with open(vocab_path, "w", encoding="utf-8") as fpath:
		for line in vocab_words:
			fpath.write(line+"\n")