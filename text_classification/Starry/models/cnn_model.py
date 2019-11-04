# encoding:utf8

from torch import nn
import torch
import torch.nn.functional as F


class CNNModel(nn.Module):
    # 在init中定义模型的零件
    def __init__(self, sentence_length, input_dim, output_dim, n_filters=100, dropout=0.5):
        # 传进来的参数： 句子的长度， 输入维度， 输出维度， CNN的层数
        super(CNNModel, self).__init__()
        self.sentence_length = sentence_length
        self.embedding = nn.Embedding(sentence_length, input_dim)
        # 定义CNN的结构。  传进去的参数是四维
        self.convs_3 = nn.Conv2d(1, n_filters, (1, input_dim))
        self.convs_4 = nn.Conv2d(1, n_filters, (1, input_dim))
        self.convs_5 = nn.Conv2d(1, n_filters, (1, input_dim))
        # 定义线性变换 全连接层
        self.fc = nn.Linear(3 * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence):
        # 传进来的参数是转换成数字的列表
        # 将其转换成词向量
        features = self.embedding(sentence).unsqueeze(1)  # ( batch_size, 1 (in_channel) , seq_len , embed_dim )
        multi_roads = []
        multi_roads.append(F.relu(self.convs_3(features)).squeeze(3))
        multi_roads.append(F.relu(self.convs_4(features)).squeeze(3))
        multi_roads.append(F.relu(self.convs_5(features)).squeeze(3))
        # 将参数传递到卷积层中  并进行池化操作
        features = [F.max_pool1d(road, road.size(2)).squeeze(2) for road in multi_roads]  # ( batch_size , out_channel)
        # 将三次卷积得到的结果进行拼接。
        concated = torch.cat(features, 1)  # ( batch_size , out_channel*len(kernal_sizes)  )
        # if is_training:
        concated = self.dropout(concated)
        # 线性变换
        output = self.fc(concated)
        # softmax函数  将输出转换成概率事件
        return F.log_softmax(output, 1)
