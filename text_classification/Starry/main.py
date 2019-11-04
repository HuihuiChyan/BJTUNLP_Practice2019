import data
from models.cnn_model import CNNModel
import torch
from torch import nn
import random
from sklearn.metrics import f1_score, accuracy_score


def averagenum(num1):
    nsum = 0
    for i in range(len(num1)):
        nsum += num1[i]
    return nsum / len(num1)


def train():
    # global epoch
    epochs = 500
    embeded_dim = 100
    lr = 0.0001
    batch_size = 1000
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    loss_func = nn.CrossEntropyLoss()
    word_index_dic, index_word_dic = data.word_index()
    length = len(word_index_dic)
    print(length)
    cnn_model1 = CNNModel(int(length), int(embeded_dim), 2)
    if use_cuda:
        cnn_model1.cuda()
    optimizer = torch.optim.Adam(cnn_model1.parameters(), lr=lr)
    train_size = 100
    result_list = []
    targer_list = []
    num = 0
    F = 0
    for epoch in range(epochs):
        for data_list in get_data_list(train_size):
            for data_item in data_list:

                input_tensor = []
                for item in data_item[0].lower().replace(".", "").replace(",", "").replace("<br", "") \
                        .replace("/>", "").replace("?", "").replace(";", "").strip().split(" "):
                    try:
                        input_tensor.append(word_index_dic[item])
                    except KeyError:
                        # print(item)
                        pass
                try:
                    if len(input_tensor) > 400:
                        input_tensor = input_tensor[:400]
                    else:
                        while len(input_tensor) < 400:
                            input_tensor = input_tensor.append(0)
                except:
                    continue
                input_tensor = torch.LongTensor([input_tensor])
                targer = torch.LongTensor([data_item[1]])
                targer_list.append(data_item[1])
                if use_cuda:
                    input_tensor = input_tensor.cuda()
                    targer = targer.cuda()
                cnn_model1.zero_grad()

                predict = cnn_model1(input_tensor)
                # print(predict.argmax())
                # print(int(predict.argmax()))
                # print(targer)
                result_list.append(int(predict.argmax()))
                loss = loss_func(predict, targer)
                # print(loss)
                loss.backward()
                optimizer.step()
                num += 1
                # if num % batch_size == 0:
                #     num = 0
                #     new_f1 = f1_score(result_list, targer_list, average='weighted')
                #     print(new_f1)
                #     # if new_f1 > F:
                #     #     F = new_f1
                #     #     print("F1:\t" + str(F) + "\n")
                #     #     parmeter = cnn_model1.state_dict()
                #     #     torch.save(parmeter, "./checkpoints_new3/CNN_model_" + str(epoch + 1) + "_f1_" + str(F) + ".pt")
                #     result_list = []
                #     targer_list = []
        parmeter = cnn_model1.state_dict()
        torch.save(parmeter, "./checkpoints_new4/CNN_model_" + str(epoch + 1) + ".pt")


def get_data_list(bach_size):
    data_list = []
    num = 0
    sen_neg = data.read_neg_data_test()
    sen_pos = data.read_pos_data_test()
    tag1 = True
    tag2 = True
    while tag1 or tag2:
        try:
            s1 = sen_neg.__next__()
            data_list.append([s1, 0])
            num += 1
        except StopIteration:
            tag1 = False
        try:
            s2 = sen_pos.__next__()
            data_list.append([s2, 1])
            num += 1
        except StopIteration:
            tag2 = False
        if tag1 is False and tag2 is False:
            return data_list
        if num > bach_size:
            num = 0
            yield data_list
            data_list = []


def test():
    import os
    print("开始测试")
    # 定义使用的显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num = 0
    number = 0
    use_cuda = torch.cuda.is_available()
    word_index_dic, index_word_dic = data.word_index()
    length = len(word_index_dic)
    embeded_dim = 100
    bach_size = 100
    torch.cuda.set_device(0)
    # 创建模型类
    cnn_model1 = CNNModel(int(length), int(embeded_dim), 2)
    if use_cuda:
        # 将模型转换成GPU版本
        cnn_model1.cuda()
    # 加载模型中的参数
    # print(list(os.walk("./checkpoints_new2")))
    # for model_file in list(os.walk("./checkpoints_new2"))[0][2]:
    #     print(model_file)
    #     path = os.path.join("./checkpoints_new1",model_file)
    # 176
    # P: 0.8079631852741096
    # F: 0.8076630195182558
    # for index in range(150, 500):

    for index in range(151, 152):
        result_list = []
        targer_list = []
        path = "./checkpoints_new/CNN_model_" \
               + str(index) + ".pt"
        print("加载模型")
        print("CNN_model_" + str(index) + ".pt")
        try:
            cnn_model1.load_state_dict(torch.load(path))
        except:
            print("error")
        # 开始测试
        for data_list in get_data_list(bach_size):
            for data_item in data_list:
                # try:
                input_tensor = []
                # 清除一些没用的字符  包括英文的逗号等
                for item in data_item[0].lower().replace(".", "").replace(",", "").replace("<br", "") \
                        .replace("/>", "").replace("?", "").replace(";", "").strip().split(" "):
                    try:
                        # 将单词映射成数字
                        key = word_index_dic[item]
                        input_tensor.append(key)
                    except KeyError:
                        pass

                if len(input_tensor) > 400:
                    input_tensor = input_tensor[:400]
                else:
                    # print(input_tensor)
                    # print(data_item[0])
                    # print(type(input_tensor))
                    while len(input_tensor) < 400:
                        input_tensor.append(0)
                # 将输入转换成tensor的格式
                input_tensor = torch.LongTensor([input_tensor])
                # 加载计算图
                with torch.no_grad():
                    if use_cuda:
                        input_tensor = input_tensor.cuda()
                    # 将输入放到模型中  获取输出预测值
                    predict = cnn_model1(input_tensor)
                    # argmax()函数用来获取  列表数值最大的数字的  “位置”
                    predict_label = int(predict.argmax())
                # print("predict_label:", predict_label, "True:", data_item[1])
                # 把预测结果和正确的结果记录下来
                result_list.append(int(predict_label))
                targer_list.append(data_item[1])
                # number += 1
                # except TypeError:
                #     num+=1
                # print(data_item)
        new_f1 = f1_score(result_list, targer_list, average='binary')
        P = accuracy_score(result_list, targer_list, )
        # with open("result_file_end.txt", "a+", encoding='utf8') as f:
        #     f.write(str(index) + "\n")
        #     f.write("P:\t" + str(P) + "\n")
        #     f.write("F:\t" + str(new_f1) + "\n")
        #     f.write("________________________________________________\n")
        # print("N:", index)
        print("P:", P)
        print("F:", new_f1)
        print("==================================")
        # print(number)
        # print(num)


if __name__ == '__main__':
    # train()
    test()
