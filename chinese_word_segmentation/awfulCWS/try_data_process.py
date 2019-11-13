# coding: utf-8

import data_utlis
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.data.dataloader as dataloader
import model
import evaluate
import numpy as np
from model import Bilstm


# 路径
'''
train_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/msr_training/msr_training.utf8"
train_test_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/msr_training/_training.utf8"
vocab_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/vocab_all.txt"
test_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/msr_testing/msr_test.utf8"
gold_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/msr_gold/msr_test_gold.utf8"
train_data_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/after_process_data/train_data.npy"
test_data_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/after_process_data/test_data.npy"
train_label_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/after_process_data/train_labels.npy"
test_label_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/after_process_data/test_labels.npy"
mini_data_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/mini_data/"
mini_train_path = mini_data_path + "mini_msr_training.utf8"
mini_test_path = mini_data_path + "mini_msr_test.utf8"
mini_gold_path = mini_data_path + "mini_msr_gold.utf8"
vocab_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/vocab.txt"
vocab = data_utlis.read_dict(vocab_path)
'''
train_path = "/home1/ljr/practice/cws/data/_training.utf8"
vocab_path = "/home1/ljr/practice/cws/data/vocab.txt"
test_path = "/home1/ljr/practice/cws/data/msr_test.utf8"
gold_path = "/home1/ljr/practice/cws/data/msr_test_gold.utf8"
mini_train_path = "/home1/ljr/practice/cws/mini_data/mini_msr_training.utf8"
mini_test_path = "/home1/ljr/practice/cws/mini_data/mini_msr_test.utf8"
mini_gold_path = "/home1/ljr/practice/cws/mini_data/mini_msr_gold.utf8"

# 数据处理

train_features,train_labels= data_utlis.generate_corpus(vocab_path,train_path)
test_features,test_labels = data_utlis.generate_corpus(vocab_path,gold_path)
'''
train_features,train_labels= data_utlis.generate_corpus(vocab_path,mini_train_path)
test_features,test_labels = data_utlis.generate_corpus(vocab_path,mini_gold_path)
'''
train_length = list(map(lambda t: len(t) + 1, train_features))
test_length = list(map(lambda t: len(t) + 1, test_features))


train_set = [[feature,label,length] for feature,label,length in zip(train_features,train_labels,train_length)]
test_set = [[feature,label,length] for feature,label,length in zip(test_features,test_labels,test_length)]


train_set.sort(key = lambda x:len(x[0]),reverse=True)
test_set.sort(key = lambda x:len(x[0]),reverse=True)


maxlen = max(len(train_set[0][0]),len(test_set[0][0]))
for i in range(0,len(train_set)):
    while(len(train_set[i][0])!=maxlen):
        train_set[i][0].append(0)
        train_set[i][1].append(0)


for i in range(0,len(test_set)):
    while(len(test_set[i][0])!=maxlen):
        test_set[i][0].append(0)
        test_set[i][1].append(0)


'''
import numpy as np
train_set_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/train_set.npy"
test_set_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/test_set.npy"
np.save(train_set_path,train_set)
np.save(test_set_path,test_set)
'''

train_features = [feature[0] for feature in train_set]
train_labels = [feature[1] for feature in train_set ]
train_lengths = [feature[2] for feature in train_set]
test_features = [feature[0] for feature in test_set]
test_labels = [feature[1] for feature in test_set]
test_lengths = [feature[2] for feature in test_set]


for i in range(1,len(train_labels)):
    if len(train_labels[i]) != maxlen:
        train_labels[i] = train_labels[i][:maxlen]

for i in range(1,len(test_labels)):
    if len(test_labels[i]) != maxlen:
        test_labels[i] = test_labels[i][:maxlen]






batch_size = 32
START_TAG = "<START>"
STOP_TAG = "<STOP>"
hidden_size = 300
embedding_dim = 300
num_layers = 1
class_nums = 6
vocab = data_utlis.read_dict(vocab_path)
vocab_size = len(vocab)
tag_to_idx = {"B": 0, "E": 1, "M": 2, "S": 3,START_TAG: 4, STOP_TAG: 5}
idx_to_tag = {0:"B",1:"I",2:"I",3:"B"}
epoch_size = 50
learning_rate = 0.05
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


train_features = torch.tensor(train_features,device=device)
train_labels = torch.tensor(train_labels,device=device)
train_lengths = torch.tensor(train_lengths,device=device)
test_features = torch.tensor(test_features,device=device)
test_labels = torch.tensor(test_labels,device=device)
test_lengths = torch.tensor(test_lengths,device=device)


train_set = torch.utils.data.TensorDataset(train_features, train_labels,train_lengths)
test_set = torch.utils.data.TensorDataset(test_features, test_labels,test_lengths)
train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False)




net = Bilstm(vocab_size,maxlen,hidden_size,embedding_dim,num_layers,class_nums,tag_to_idx,batch_size,device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-4)
net.to(device)


'''
res_file = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/result"
temp_file = "D:\\study\\bjtu\\nlp_practice\\perl\\eg\\tmp"
conneval_path = "D:\\study\\bjtu\\nlp_practice\\perl\\eg\\conlleval.pl"
log_path = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/log"
'''

#res_file = "D:/study/bjtu/nlp_practice/chinese_word_segmentation/data/result"
temp_file = "/home1/ljr/practice/cws/tmp"
conneval_path = "/home1/ljr/practice/cws/conlleval.pl"
log_path = "/home1/ljr/practice/cws/log"

count_step = 0

for epoch in range(epoch_size):
    for feature,label,length in train_iter:
        count_step  += 1
        net.zero_grad()
        feature = feature.to(device)
        label = label.to(device)
        loss = net.neg_log_likelihood(feature, label)
        loss.backward()
        optimizer.step()
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        if(count_step%30 == 0):
            with torch.no_grad():
                temp_file = data_utlis.predict_write(net,test_iter,temp_file)
                data_utlis.evaluate_perl(conneval_path,temp_file,count_step//30,log_path)

        #梯度裁剪
    #调整lr
    #测试，将测试结果输入到文件，按照per格式
    #对测试文件进行测试，保存测试结果（1 epoch的结果
    '''
    with torch.no_grad():
        temp_file = data_utlis.predict_write(net,test_iter,temp_file)
        data_utlis.evaluate_perl(conneval_path,temp_file,epoch+1,log_path)
    ...

    
    

