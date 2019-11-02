import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def remove_(text):
	punctuation = '\[\]\{\}\(\)!,.;:?"\''
	text = re.sub(r'[{}]+'.format(punctuation),'',text)
	return text.strip().lower()

def remove(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower() 

def read_file(path):
	texts = []
	labels = []
	f = open(path)
	print('Reading file...')
	num = 0
	for line in tqdm(f):
		if num == 0:
			num += 1
			continue
		item = line.strip().split('\t')
		text_a, text_b = remove(item[5]), remove(item[6])
		label = [0, 0, 0]
		label[int(item[0])] = 1
		labels.append(label)
		texts.append([text_a, text_b])
	return texts, labels

def read_vocab(vocab_dir='./vocab.txt'):
	dic = {}
	f = open(vocab_dir, 'r')
	num = 0
	for line in f:
		item = line.strip()
		dic[item] = num
		num += 1
	print('vocab_size:', len(dic))
	return dic

def sent2vec(sents, vocab, max_len):
	vecs = []
	print('Converting to index ...')
	for sent_a, sent_b in tqdm(sents):
		vec_a = [vocab[word] if word in vocab else 0 for word in sent_a.split(' ')]
		if len(vec_a) <= max_len:
			vec_a += [0 for _ in range(max_len - len(vec_a))]
		else:
			vec_a = vec_a[:max_len]

		vec_b = [vocab[word] if word in vocab else 0 for word in sent_b.split(' ')]		
		if len(vec_b) <= max_len:
			vec_b += [0 for _ in range(max_len - len(vec_b))]
		else:
			vec_b = vec_b[:max_len]
		vecs.append([vec_a, vec_b])
	return vecs

def generator(data, batch_size):
	a_, b_, labels = [], [], []
	for (a, b), label in data:
		if len(a_) == batch_size:
			yield a_, b_, labels
			a_, b_, labels = [], [], []
		a_.append(a)
		b_.append(b)
		labels.append(label)
	if len(a_) != 0:
		yield a_, b_, labels

