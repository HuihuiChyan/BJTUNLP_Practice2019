# encoding:utf8

import os

from config import Config


def read_pos_data():
    path = Config.data_path + "train/pos/"
    for item in list(os.walk(path))[0][2]:
        path1 = os.path.join(path, item)
        # print(path1)
        try:
            with open(path1, "r") as f:
                yield f.read()
        except UnicodeDecodeError:
            with open(path1, "r", encoding="utf8") as f:
                yield f.read()


def read_neg_data():
    path = Config.data_path + "train/neg/"
    for item in list(os.walk(path))[0][2]:
        path1 = os.path.join(path, item)
        # print(path1)
        try:
            with open(path1, "r") as f:
                yield f.read()
        except UnicodeDecodeError:
            with open(path1, "r", encoding="utf8") as f:
                yield f.read()


def read_pos_data_test():
    path = Config.data_path + "test/pos/"
    for item in list(os.walk(path))[0][2]:
        path1 = os.path.join(path, item)
        # print(path1)
        try:
            with open(path1, "r") as f:
                yield f.read()
        except UnicodeDecodeError:
            with open(path1, "r", encoding="utf8") as f:
                yield f.read()


def read_neg_data_test():
    path = Config.data_path + "test/neg/"
    for item in list(os.walk(path))[0][2]:
        path1 = os.path.join(path, item)
        # print(path1)
        try:
            with open(path1, "r") as f:
                yield f.read()  
        except UnicodeDecodeError:
            with open(path1, "r", encoding="utf8") as f:
                yield f.read()


def word_index():
    word_index_dic = {}
    index_word_dic = {}
    with open("/home1/huanghui/dataset4practice/text_classification/aclImdb_v1/imdb.vocab","r",encoding="utf8") as f:
        # print(f)
        for index, item in enumerate(f.readlines()):
            # print(index)
            # print(item.strip())
            word_index_dic[item.strip()] = index
            index_word_dic[index] = item.strip()
    return [word_index_dic, index_word_dic]


if __name__ == '__main__':
    reader = read_neg_data()
    for i in reader:
        print(i)
