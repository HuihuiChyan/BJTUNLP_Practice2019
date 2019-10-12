import re 
import os
import nltk
import numpy as np
from itertools import chain
from nltk.corpus import stopwords





def read_files(path,filetype):
	file_list = []
	pos_path = path + filetype + "/pos/"
	neg_path = path + filetype +"/neg/"
	for f in os.listdir(pos_path):
		file_list += [[pos_path+f,1]]
	for f in os.listdir(neg_path):
		file_list +=[[neg_path+f,0]]
	data = []
	for fi,label in file_list:
		with open(fi,encoding='utf8') as fi:
			data += [[" ".join(fi.readlines()),label]]
	return data

def get_stop_words_list(filepath):
	stop_words_list = []
	with open(filepath,encoding='utf8') as fi:
		for line in fi.readlines():
			stop_words_list.append(line.strip())
	return stop_words_list


def data_process(text):
	re_tag = re.compile(r'[^[a-z\s]')
	text.lower()
	text = re_tag.sub('',text)
	#text = " ".join([word for word in text.split(' ') if word not in stopwords.words('english')])
	text = " ".join([word for word in text.split(' ')])
	return text

def get_token_text(text):
	#token_data = [data_process(st) for st in text.split()]
	token_data = [st.lower() for st in text.split()]
	token_data = list(filter(None,token_data))
	return token_data

def get_token_data(data):
	data_token = []
	for st,label in data:
		data_token.append(get_token_text(st))
	return data_token

def get_vocab(data):
	vocab = set(chain(*data))
	vocab_size = len(vocab)
	word_to_idx  = {word: i+1 for i, word in enumerate(vocab)}
	word_to_idx['<unk>'] = 0
	idx_to_word = {i+1: word for i, word in enumerate(vocab)}
	idx_to_word[0] = '<unk>'
	return vocab,vocab_size,word_to_idx,idx_to_word

def encode_st(token_data,vocab,word_to_idx):
	features = []
	for sample in token_data:
		feature = []
		for token in sample:
			if token in word_to_idx:
				feature.append(word_to_idx[token])
			else:
				feature.append(0)
		features.append(feature)
	return features

def pad_st(features,maxlen,pad=0):
	padded_features = []
	for feature in features:
		if len(feature)>maxlen:
			padded_feature = feature[:maxlen]
		else:
			padded_feature = feature
			while(len(padded_feature)<maxlen):
				padded_feature.append(pad)
		padded_features.append(padded_feature)
	return padded_features

def read_file(path):
	a = np.load(path,allow_pickle=True)
	a = a.tolist()
	return a




	





