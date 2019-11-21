# encoding=utf-8
import numpy as np
import re


# 过滤函数
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", "\'s", string)
    string = re.sub(r"\'ve", "\'ve", string)
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


# 加载数据的函数
def load_data(path):
    texts = []
    labels = []
    file = open(path, 'r', encoding='utf-8')
    print('Reading file...')
    for line in file:
        label = line.strip().split('\t')[0]
        text = clean_str(line.strip().split('\t')[1])
        labels.append([1-float(label), float(label)])
        texts.append(text)
    return texts, labels


# 读词表
def read_vocab(vocab_dir='./vocab.txt'):
    dic = {}
    file = open(vocab_dir, 'r', encoding='utf-8')
    num = 0
    for line in file:
        item = line.strip()
        dic[item] = num
        num += 1
    print('vocab_size:', len(dic))
    return dic


# 将句子转为下标
def sent2idx(sents, vocab, max_len):
    idxs = []
    print('Converting to index...')
    for sent in sents:
        idx = []
        idx = [vocab[word] if word in vocab else 0 for word in sent.split(' ')]
        # 不够max_len长度补0，超出的截掉
        if len(idx) <= max_len:
            idx += [0 for _ in range(max_len-len(idx))]
        else:
            idx = idx[:max_len]
        idxs.append(idx)
    return idxs

"""
# 生成batch数据
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    # num_epochs是干嘛用的
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]
"""


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