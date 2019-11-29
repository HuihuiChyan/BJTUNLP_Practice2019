import re
from functools import reduce
from tensorflow.keras import preprocessing
import numpy as np
import gensim

def build_vocab():
	w = open('data/vocab.txt','w', encoding='utf-8')
	vocab = set()
	for line in open('data/msr_training/msr_training.utf8'):
		string = ''.join(line.strip().split(' '))
		for i in string:
			if i in vocab:
				continue
			else:
				w.write(i+'\n')
				vocab.add(i)

# build_vocab()

def convert_corpus_label(sentence):
    """
    :param sentence: 类似于 '戈壁 上 ， 也 并不是 寸草不生'
    :return: ['B','E','S','S','S','B','M','E','B','M','M','E']
    """
    label = []
    sentence = re.sub('   ', ' ', sentence) # 将三空格 转变为单空格
    sentence = re.sub('  ', ' ', sentence)  # 将双空格 转变为单空格
    l = sentence.split(' ')
    s = ''
    for item in l:
        s += item.strip()
        if len(item) == 1:
            label.append('S')
        elif len(item) == 2:
            label.append('B')
            label.append('E')
        else:
            label.append('B')
            label.extend('M' * (len(item) - 2))
            label.append('E')
    return s, label

def get_input_label_file(filename):
    """
    return : 
        train.utf8 : sentences '中文 分词'
        label.utf8 : labels     'B E B E'
    """
    with open(filename, 'r', encoding='utf-8') as fr:
        content = fr.readlines()
    corp = open('data/train.utf8', 'w', encoding='utf-8')
    label = open('data/label.utf8', 'w', encoding='utf-8')

    corp_s = ''
    label_s = ''
    for s in content:
        s = s.strip()
        raw_s, raw_l = convert_corpus_label(s)
        for c in raw_s:
            corp_s += c + ' '
        for l in raw_l:
            label_s += l + ' '
        corp_s += '\n'
        label_s += '\n'
    corp.write(corp_s)
    label.write(label_s)
    
# get_input_label_file('data/msr_training/msr_training.utf8')

def tokenize(filename, is_label=False):
    # 创建tokenizer 字符器
    with open(filename, 'r', encoding='utf-8') as fr:
        content = fr.readlines()
    content = list(map(lambda sentence: sentence.strip(), content))

    # 中文输入tokenizer
    if is_label == False:
        tokenizer = preprocessing.text.Tokenizer(num_words=6000, filters='', oov_token='<unk>')
    else:
        tokenizer = preprocessing.text.Tokenizer(num_words=6000, filters='')
    tokenizer.fit_on_texts(content)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>' 

    sequences = tokenizer.texts_to_sequences(content)
    sequences = preprocessing.sequence.pad_sequences(sequences, padding='post')

    return sequences, tokenizer

# _, text_tokenizer = tokenize('data/train.utf8')
# __, label_tokenizer = tokenize('data/label.utf8', True)
# print(len(text_tokenizer.word_index))
# print(_[:3])
# print(label_tokenizer.word_index)
# print(__[:3])
# print(text_tokenizer.sequences_to_texts(_[:3]))
# print(label_tokenizer.sequences_to_texts(__[:3]))

def preprocess_test():
    with open('../data/msr_testing/msr_test.utf8', 'r', encoding='utf-8') as fr:
        content = fr.readlines()
    text = ''
    for s in content:
        s = s.strip()
        for c in s:
            text += c + ' '
        text += '\n'
    with open('../data/test.utf8', 'w', encoding='utf-8') as fw:
        fw.write(text)

# preprocess_test()

def convert_gold_texts2sequences(tokenizer, gold_filename):
    with open(gold_filename, 'r', encoding='utf-8') as fr:
        content = fr.readlines()
    gold_labels = []
    for sentence in content:
        sentence = sentence.strip()
        s, labels = convert_corpus_label(sentence)
        gold_labels.append(labels)
    
    gold_labels = tokenizer.texts_to_sequences(gold_labels)

    return gold_labels

# __, label_tokenizer = tokenize('../data/label.utf8', True)
# gold_labels = convert_gold_texts2sequences(label_tokenizer, '../data/msr_gold/msr_test_gold.utf8')
# print(gold_labels[:10])

def get_pretrain_table(pretrain_table):
    table = gensim.models.KeyedVectors.load_word2vec_format(pretrain_table, binary=True, unicode_errors='ignore')
    return table

table = get_pretrain_table('../data/token_vec_300.bin')
for i in range(5):
    print(table[])

    

