import codecs
from tqdm import tqdm
from hparams import Hparams as hp
import random
import numpy as np



def load_vocab(vocab_path, vocab_size):
    vocab = [line.split()[0] for line in codecs.open(vocab_path, 'r', encoding='utf-8').read().splitlines()]
    vocab = vocab[:vocab_size]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_data(data_path, name=None):
    data = []
    with codecs.open(data_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    _sent, _tag = [], []
    for line in tqdm(lines):
        if line != '\n':
            [char, label] = line.strip().split()
            _sent.append(char)
            _tag.append(label)
        else:
            if len(_sent) <= hp.max_len:
                data.append((_sent, _tag))
                _sent, _tag = [], []
    print("\nLoad %s data over. sentences:%d\n" %(name, len(data)))
    return data


def padding(x_list, y_list, maxlen):
    X = np.zeros([len(x_list), maxlen], np.int32)
    Y = np.zeros([len(y_list), maxlen], np.int32)
    seqs_len = np.zeros([len(x_list)], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, maxlen-len(y)], 'constant', constant_values=(0, 0))
        seqs_len[i] = min(len(x), hp.max_len)
    return X, Y, seqs_len


def get_batch(data, batch_size, vocab_path, tag2label, shuffle=False):
    if shuffle:
        random.shuffle(data)

    word2idx, _ = load_vocab(vocab_path, hp.vocab_size)

    seqs, labels = [], []
    for (_sent, _tag) in data:
        sent_id = []
        # if len(_sent) > hp.max_len: continue
        for word in _sent:
            word_id = word2idx.get(word, 2)
            sent_id.append(word_id)
        label_id = [tag2label[tag] for tag in _tag]
        if len(seqs) == batch_size:
            seqs, labels, seqs_len = padding(seqs, labels, hp.max_len)
            yield seqs, labels, seqs_len
            seqs, labels = [], []
        seqs.append(sent_id)
        labels.append(label_id)

    if len(seqs) != 0:
        seqs, labels, seqs_len = padding(seqs, labels, hp.max_len)
        yield seqs, labels, seqs_len


if __name__ == '__main__':
    data = load_data(hp.prepro_dir + 'test.utf8')
    for (sent, tag) in data:
        print(sent)
        print(tag)
    for seqs, labels, seqs_len in get_batch(data, hp.batch_size, hp.vocab_path, hp.tag2label, shuffle=False):
        print(seqs)
        print(labels)
        print(seqs_len)
        exit(1)
