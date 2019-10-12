from model_CNN import textCNN
from data_utlis import read_file
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import time
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score


embed_size = 300
num_hidens = 100
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.05
device = torch.device('cuda:1')
use_gpu = True
seq_len = 300
droput = 0.05
num_epochs = 1000

train_features_path = "/home1/ljr/practice/text_classification/textCNN_pytorch/data/test_features.npy"
test_features_path = "/home1/ljr/practice/text_classification/textCNN_pytorch/data/test_features.npy"
test_label_path = "/home1/ljr/practice/text_classification/textCNN_pytorch/data/test_label.npy"
train_label_path = "/home1/ljr/practice/text_classification/textCNN_pytorch/data/train_label.npy"
vocab_path = "/home1/ljr/practice/text_classification/textCNN_pytorch/data/vocab.npy"

word2vec_path = "/home1/ljr/practice/text_classification/glove_to_word2vec_300d.txt"
output_path = "/home1/ljr/practice/text_classification/textCNN_pytorch/result.txt"

train_features = read_file(train_features_path)
test_features = read_file(test_features_path)
train_label = read_file(train_label_path)
test_label = read_file(test_label_path)
vocab = read_file(vocab_path)
vocab_size = len(vocab)

train_features = torch.tensor(train_features)
test_features = torch.tensor(test_features)
train_labels = torch.tensor(train_label)
test_labels = torch.tensor(test_label)

wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path,binary=False, encoding='utf-8')
weight = torch.zeros(vocab_size+1, embed_size)
for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmode.index2word[i]]
    except:
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(
        idx_to_word[word_to_idx[wvmodel.index2word[i]]]))

net = textCNN(vocab_size=(vocab_size+1), embed_size=embed_size , seq_len= seq_len, labels=labels, weight=weight,droput=droput)
net.to(device)

train_set = torch.utils.data.TensorDataset(train_features, train_labels)
test_set = torch.utils.data.TensorDataset(test_features, test_labels)
train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer,100,0.5)

f=open(output_path,"w")
best_test_acc  = 0

for epoch in range(num_epochs):
    scheduler.step()
    start = time.time()
    train_loss, test_losses = 0, 0
    train_acc, test_acc = 0, 0
    n, m = 0, 0
    for feature, label in train_iter:
        n += 1
        net.zero_grad()
        feature = feature.to(device)
        label = label.to(device)
        score = net(feature)
        loss = loss_function(score, label)
        loss.backward()
        optimizer.step()
        train_acc += accuracy_score(torch.argmax(score.cpu().data,dim=1), label.cpu())
        train_loss += loss
    with torch.no_grad():
        for test_feature, test_label in test_iter:
            m += 1
            test_feature = test_feature.to(device)
            test_label = test_label.to(device)
            test_score = net(test_feature)
            test_loss = loss_function(test_score, test_label)
            test_acc += accuracy_score(torch.argmax(test_score.cpu().data,dim=1), test_label.cpu())
            test_losses += test_loss
    end = time.time()
    runtime = end - start
    if(test_acc>best_test_acc):
    	best_test_acc = test_acc
    f.write('epoch: %d, train loss: %.4f, train acc: %.5f, test loss: %.4f, test acc: %.5f, best test acc: %.5f,time: %.4f \n' %(epoch, train_loss.data / n, train_acc / n, test_losses.data / m, test_acc / m, best_test_acc / m,runtime))
    print('epoch: %d, train loss: %.4f, train acc: %.5f, test loss: %.4f, test acc: %.5f, best test acc: %.5f,time: %.4f' %(epoch, train_loss.data / n, train_acc / n, test_losses.data / m, test_acc / m, best_test_acc / m,runtime))
