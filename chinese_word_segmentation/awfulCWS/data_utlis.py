# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import torch.nn as nn
from model import Bilstm
START_TAG = "<START>"
STOP_TAG = "<STOP>"




train_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/msr_training/msr_training.utf8"
test_path = "test_path = D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/msr_testing/msr_test.utf8"
vocab_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/vocab_all.txt"
tag_to_idx = {"B": 0, "E": 1, "M": 2, "S": 3,START_TAG: 4, STOP_TAG: 5}
idx_to_tag = {0:"B",1:"I",2:"I",3:"B"}


#加载保存的向量文件
def read_data_file(path):
	a = np.load(path,allow_pickle=True)
	a = a.tolist()
	return a

#读取词典
def read_dict(path):
	#{0：[PAD],1:w1,2:w2...n:wn}
	vocab = dict()
	with open(path,encoding='utf-8') as f:
		data = f.read().split('\n')[:-1]
	for idx,word in enumerate(data):
		vocab[idx+1] = word
	vocab[0] = "[PAD]"
	return vocab

#读取文件 按行分割成列表
def read_file(path):
	#data['sent1','sent2'...'sentn']
	with open(path,encoding='utf-8') as f:
		data = f.read().split('\n')
	return data
#对一句话进行标注
def get_sent_label(sent):
	label = []
	for word in sent:
		if word == '[UNK]':
			label.append('S')
		elif len(word) == 1:
			label.append('S')
		elif len(word) == 2:
			label.append('B')
			label.append('E')
		else:
			label.append('B')
			for i in range(len(word)-2):
				label.append('M')
			label.append('E')
	return label

#将一句话按分词结果切割
def get_ch_data(data,maxlen=50):
	ch_data = []
	for sent in data:
		temp_data  = sent.strip(' ').split('  ')
		#if len(temp_data)<maxlen:
		#	for i in range(maxlen - len(temp_data)):
		#		temp_data.append("[UNK]")
		ch_data.append(temp_data)
	return ch_data

#得到label集
def get_label_data(ch_data,maxlen=50):
	label = []
	for sent in ch_data:
		temp_label = get_sent_label(sent)
		label.append(temp_label)
	return label

#得到idx——label集
def get_idx_label(label_data):
	idx_label = []
	for label_ in label_data :
		temp_idx_data = []
		for label in label_:
			temp_idx_data.append(tag_to_idx[label])
		idx_label.append(temp_idx_data)
	return idx_label


#sent2id
def get_sent_id(sent,vocab_path,maxlen=50):
	vocab = read_dict(vocab_path)
	vocab = {value:key for key,value in vocab.items()}
	sent2id = []
	for word in sent:
		sent2id.append(vocab.get(word,1))
	return sent2id


#得到idx数据集
def get_id_data(vocab_path,data_path,maxlen=50):
	vocab = read_dict(vocab_path)
	data = read_file(data_path)
	#ch_data = get_ch_data(data)
	id_data = []
	for sent in data:
		temp_sent = []
		for word in sent:
			if word !=' ':
				temp_sent.append(word)
		id_data.append(get_sent_id(temp_sent,vocab_path,maxlen))
	return id_data

def generate_corpus(vocab_path,data_path):
	features = get_id_data(vocab_path,data_path)
	data = read_file(data_path)
	ch_data = get_ch_data(data)
	label = get_label_data(ch_data)
	labels = get_idx_label(label)
	return features,labels

#结果写入预测函数
def predict_write(model,test_iter,tmp_file='./tmp'):
	with open(tmp_file,"w",encoding='utf8') as f:
		for feature,label,length in test_iter:
			score,predict_label = model(feature)
			for one_feature,one_label,one_pre,one_length in zip(feature,label,predict_label,length):
				for ch_f,ch_c,ch_p in zip(one_feature[:one_length],one_label[:one_length],one_pre[:one_length]):
					temp_output = str(int(ch_f)) + ' ' + idx_to_tag[int(ch_c)] + ' ' + idx_to_tag[ch_p] + '\n'
					f.write(temp_output)
	return tmp_file

def evaluate_perl(conll_file,pre_file,epoch,log,result_file='./result'):
	os.system("perl %s < %s > %s"%(conll_file,pre_file,result_file))
	with open(result_file,'r',encoding='utf8') as f:
		lines = f.readlines()
	with open(log,'a',encoding='utf8') as f :
		f.write('----------------epoch : %d test result----------------\n'%(epoch))
		f.write(lines[0])
		f.write(lines[1])
	




	




