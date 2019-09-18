import numpy as np
import re
import os
import collections
from operator import itemgetter
import pandas as pd
import tensorflow.keras as keras

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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


def load_data_and_labels(data_file, labels_file):
    """
    !!!notes:!!! something have been wrong with data processing
    using load_data_and_labels_new(data_file) instead !!!

    have already clean_str
    return: x = [[12 991 10],[183,65,...,326],[69 9 1 14 93 449']
            y = [1,4,9,10,3,4,8,...,rate_n]
    """
    # Load data from files
    with open('dataset/imdb.vocab', 'r', encoding='utf-8') as fr:
        vocab = [i.strip() for i in fr.readlines()]
    vocab_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    with open(data_file, 'r', encoding='utf-8') as fr:
        x = [clean_str(s) for s in fr.readlines()]
    with open(labels_file, 'r', encoding='utf-8') as fr:
        y = fr.readlines()
    y = [int(item.strip()) for item in y]

    x_id = []
    for sen in x:
        tmp = sen.split()
        tmp_id = []
        for word in tmp:
            if word in vocab_id.keys():
                tmp_id.append(vocab_id[word])
            else:
                tmp_id.append(vocab_id['<unk>'])
        x_id.append(tmp_id)

    return np.array(x_id), np.array(y)

def load_data_and_labels_new(data_file):
    """
        have already clean_str and convert unknown word into '<unk>' which index is 1
        return: x = [[12 991 10],[183,65,...,326],[69 9 1 14 93 449']
                y = [1,4,9,10,3,4,8,...,rate_n]
        """
    # Load data from files
    with open('dataset/imdb.vocab', 'r', encoding='utf-8') as fr:
        vocab = [i.strip() for i in fr.readlines()]
    vocab_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    with open(data_file, 'r', encoding='utf-8') as fr:
        content = [s.split('\t') for s in fr.readlines()]
    x = []
    y = []
    for idx in range(len(content)):
        x.append(clean_str(content[idx][1]))
        y.append(int(content[idx][0]))

    x_id = []
    for sen in x:
        tmp = sen.split()
        tmp_id = []
        for word in tmp:
            if word in vocab_id.keys():
                tmp_id.append(vocab_id[word])
            else:
                tmp_id.append(vocab_id['<unk>'])
        x_id.append(tmp_id)

    return np.array(x_id), np.array(y)



def batch_iter(data, labal, batch_size, max_len=None):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = data[shuffle_indices]
    shuffled_labels = labal[shuffle_indices]

    for i in range(batch_size, data_size, batch_size):
        x = shuffled_data[i - batch_size:min(i, data_size)]
        y = shuffled_labels[i - batch_size:min(i, data_size)]
        x, _ = padding_sentence(x, max_len)
        yield (x, y)




def list_dir(data_dir):
    """
    :param data_dir: path of data directory
    :return: get a filename list in data_dir
    """
    # print(sys.path)
    return os.listdir(data_dir)


def concate_texts(pos_data_dir, neg_data_dir, save_dir):
    """

    not used in outside module
    :param pos_data_dir: path of positive data directory
    :param neg_data_dir: path of negative data directory
    :param save_dir: path of directory of saving data
    :return: (pos_context.txt, pos_labels.txt), (neg_context.txt, neg_labels.txt)
    """
    pos_data_list = list_dir(pos_data_dir)
    neg_data_list = list_dir(neg_data_dir)

    # read context and corresponding label
    # for example  in '24_8.txt'
    # context : 'Although this was ... A great children's story and very likable characters.'
    # label : 8
    context = []
    labels = []
    for filename in pos_data_list:
        abs_filename = pos_data_dir + '/' + filename
        with open(abs_filename, 'r', encoding='utf-8') as fr:
            context.append(fr.readline() + '\n')
        name = filename.split('_')
        rate = name[-1][0]  # get rate: 8
        labels.append(str(rate) + '\n')
    with open(save_dir + '/pos_context.txt', 'w', encoding='utf-8') as fw:
        fw.write(''.join(context))
    with open(save_dir + '/pos_labels.txt', 'w', encoding='utf-8') as fw:
        fw.write(''.join(labels))

    context = []
    labels = []
    for filename in neg_data_list:
        abs_filename = neg_data_dir + '/' + filename
        with open(abs_filename, 'r', encoding='utf-8') as fr:
            context.append(fr.readline() + '\n')
        name = filename.split('_')
        rate = name[-1][0]  # get rate: 8
        labels.append(str(rate) + '\n')
    with open(save_dir + '/neg_context.txt', 'w', encoding='utf-8') as fw:
        fw.write(''.join(context))
    with open(save_dir + '/neg_labels.txt', 'w', encoding='utf-8') as fw:
        fw.write(''.join(labels))


def statistic(data_file, labels_file):
    """
    max length of text and
    :param data_file: path of file
    :return: max length of text
    """
    x, _ = load_data_and_labels(data_file, labels_file)
    counter = collections.Counter([len(item) for item in x])
    counter = sorted(counter.items(), key=lambda d: d[0], reverse=True)
    len_seq = [item[0] for item in counter]
    # print('max length: ', counter[0][0], ', frequency: ', counter[0][1])
    d = pd.DataFrame()
    d['length'] = len_seq
    print(d.describe())
    # count   3153.000000
    # mean    2036.469077
    # std     1375.297211
    # min     51.000000
    # 25%     961.000000
    # 50%     1772.000000
    # 75%     2797.000000
    # max     8793.000000


def padding_sentence(x_batch, max_len=None):
    """
    padding sentence to the max_batch_size
    :param x_batch: a batch of a train set 'without embedding'
    :return: padded batches of sentences, original sentence length(if dynamic_rnn)
    """
    seq_len = []
    if max_len is None:
        batch_max_len = max(map(lambda x: len(x), x_batch))
    else:
        batch_max_len = max_len
    for ith_sen in x_batch:
        seq_len.append(len(ith_sen))
    padded_sentence = keras.preprocessing.sequence.pad_sequences(x_batch, batch_max_len,
                                                              padding='post', truncating='post')
    return padded_sentence, seq_len





if __name__ == '__main__':
    # pass
    # print(len(list_dir('dataset/train/pos')))
    # print(clean_str("At the beginning of the film we watch May and Toots preparing for their trip to London for a visit to their grown children. One can see Toots is not in the best of health, but he goes along. When he dies suddenly, May's world, begins to spin out of control.<br /><br />The film directed by Roger Michell, based on a screen play by Hanif Kureshi, is a study of how this mother figure comes to terms with her new status in life and her awakening into a world that she doesn't even know it existed until now.<br /><br />May's life as a suburban wife was probably boring. Obviously her sexual life was next to nothing. We get to know she's had a short extra marital affair, then nothing at all. When May loses her husband she can't go back home, so instead, she stays behind minding her grandson at her daughter's home. It is in this setting that May begins lusting after young and hunky Darren, her daughter's occasional lover.<br /><br />Darren awakes in May a passion she has not ever known. May responds by transforming herself in front of our eyes. May, who at the beginning of the film is dowdy, suddenly starts dressing up, becoming an interesting and attractive woman. She ends up falling heads over heels with this young man that keeps her sated with a passion she never felt before.<br /><br />Having known a couple of cases similar to this story, it came as no surprise to me to watch May's reaction. Her own chance of a normal relationship with Bruce, a widower, ends up frustratingly for May, who realizes how great her sex is with Darren. The younger man, we figure, is only into this affair to satisfy himself and for a possibility of extorting money from May. Finally, the daughter, Helen discovers what Mum has been doing behind her back when she discovers the erotic paintings her mother has made.<br /><br />The film is a triumph for the director. In Anne Reid, Mr. Michell has found an extraordinary actress who brings so much to the role of May. Also amazing is Daniel Craig. He knows how Darren will react to the situation. Anna Wilson Jones as Helen is also vital to the story as she is the one that has to confront the mother about what has been going on behind her back. Oliver Ford Davies plays a small part as Bruce the older man in Helen's class and is quite effective.<br /><br />The film is rewarding for those that will see it with an open mind.<br /><br />"))
    # concate_texts('dataset/test/pos', 'dataset/test/neg', 'dataset/test_dataset')
    x, y = load_data_and_labels_new('train.txt')
    print(x[:2])
    print(y[:2])

    # statistic('dataset/train_dataset/neg_context.txt', 'dataset/train_dataset/neg_labels.txt')
    # statistic('dataset/test_dataset/neg_context.txt', 'dataset/test_dataset/neg_labels.txt')
    # statistic('dataset/train_dataset/pos_context.txt', 'dataset/train_dataset/pos_labels.txt')
