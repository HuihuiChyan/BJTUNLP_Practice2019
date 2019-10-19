# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:54:14 2019

@author: dangrui
"""
import collections
import os,re
import numpy as np
"""
def clean_str(string):
    
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    
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
def load_data_and_labels(positive_data_file, negative_data_file):
    
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

"""
def read_imdb(folder='train'):
    data=[]
    labels=[]
    for label in ['pos','neg']:
        folder_name=os.path.join('./aclImdb_v1/',folder,label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name,file),'rb') as f:
                review=f.read().decode('utf-8').replace('\n',' ').lower()
                review=re.sub('<.*?>','',review,flags=re.S)
                data.append(review)
                labels.append([0,1] if label=='pos' else [1,0])
    labels=np.array(labels)
    return data,labels

(train_data,train_labels),(test_data,test_labels)=read_imdb('train'),read_imdb('test')

def get_tokenized_imdb(data):
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review in data]
#vocabulary应该为数�?
#此时vocabulary为raw dataset
#create vocab删掉次少�?的词�?
#tokenized_data=[]
def get_vocab_imdb(data):
    tokenized_data=get_tokenized_imdb(data)
    counter=collections.Counter([tk for st in tokenized_data for tk in st])
    counter=dict(filter(lambda x:x[1]>=8,counter.items()))
    return counter

MAX_LENGTH=300#define 每一个文本长�?
def get_dataset(data,length):
    max_l=length
    counter=get_vocab_imdb(data)
    #print(len(counter))
    idx_to_token=[tk for tk,_ in counter.items()]
    token_to_idx={tk:idx for idx,tk in enumerate(idx_to_token)}#token_to_idx[tk]=idx
    #print(idx_to_token[55]) revisionism
    datasets=[[token_to_idx[tk] for tk in st if tk in token_to_idx] for st in data]
    def pad(x):
        return x[:max_l] if len(x)>max_l else x+[0]*(max_l-len(x))
    return [pad(dataset) for dataset in datasets]



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
