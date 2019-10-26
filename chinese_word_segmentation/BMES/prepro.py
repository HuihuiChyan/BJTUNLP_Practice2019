# encoding=utf-8
import re
import os
import codecs
from collections import Counter
from tqdm import tqdm
from hparams import Hparams as hp

import logging
logging.basicConfig(level=logging.INFO)



def tagging(word):
    if len(word) == 1:
        return 'S'
    else:
        return 'B' + 'M'*(len(word)-2) + 'E'


def make_vocab(fin, fout, vocab_size):
    logging.info("Making vocabulary for " + fin)
    # text = codecs.open(fin, 'r', encoding='utf-8').read()
    # words = text.split()
    word2cnt = Counter()
    with codecs.open(fin, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in tqdm(lines):
        if line != '\n':
            [char, _] = line.strip().split()
            word2cnt.update(char)

    with open(fout, 'w', encoding='utf-8') as fw:
        fw.write("{}\n{}\n{}\n".format("<PAD>", "<EOS>", "<UNK>"))
        for word, cnt in word2cnt.most_common(vocab_size):
            # fw.write(u"{}\t{}\n".format(word, cnt))
            fw.write(u"{}\n".format(word))


def preprocess(fin, fout):
    with open(fin, 'r', encoding='utf-8') as fr,\
        open(fout, 'w', encoding='utf-8') as fw:
        for sent in fr:
            for word in sent.split():
                #print(word)
                tag = tagging(word)
                for i in range(len(tag)):
                    fw.write(word[i]+' '+tag[i]+'\n')
            fw.write("\n")


if __name__ == '__main__':
    if not os.path.exists(hp.prepro_dir): os.mkdir(hp.prepro_dir)
    preprocess(hp.train_path, hp.prepro_dir+'train.utf8')
    preprocess(hp.test_gold_path, hp.prepro_dir+'test.utf8')
    make_vocab(hp.prepro_dir+'train.utf8', hp.vocab_path, hp.vocab_size)
    logging.info("Done preprocessing")
