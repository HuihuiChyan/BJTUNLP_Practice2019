# encoding:utf8


class Config(object):
    data_path = "/home1/huanghui/dataset4practice/text_classification/aclImdb_v1/"
    lr = 1e-3  
    use_gpu = False
    epoch = 200
    plot_every = 20 
    max_sentence_len = 200  
    checkpoints_path = "./checkpoints"


def get():
    pass
