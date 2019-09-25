import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def remove(text):
	punctuation = '\[\]\{\}\(\)!,.;:?"\''
	text = re.sub(r'[{}]+'.format(punctuation),'',text)
	return text.strip().lower()
 

def read_file(path):
	texts = []
	labels = []
	f = open(path)
	print('Reading file...')
	for line in tqdm(f):
		label = line.strip().split('\t')[0]
		text = remove(line.strip().split('\t')[1])
		labels.append([1-float(label), float(label)])
		texts.append(text)
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
	for sent in tqdm(sents):
		vec = []
		#print(sent)
		vec = [vocab[word] if word in vocab else 0 for word in sent.split(' ')]
		#for word in sent.split(' '):
		#	if word in vocab:
		#		vec.append(vocab.index(word))
		#	else:
		#		vec.append(0)
		if len(vec) <= max_len:
			vec += [0 for _ in range(max_len - len(vec))]
		else:
			vec = vec[:max_len]
		vecs.append(vec)
	return vecs

def generator(data, batch_size):
	sents, labels = [], []
	for sent, label in data:
		if len(sents) == batch_size:
			yield sents, labels
			sents, labels = [], []
		sents.append(sent)
		labels.append(label)
	if len(sents) != 0:
		yield sents, labels

