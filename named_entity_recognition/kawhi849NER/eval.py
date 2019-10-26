# coding=utf-8

import os


def conlleval(label_predict, label_path, metric_path):
    """
    计算acc／pre／rec／F1， 需要使用到conlleval.pl
    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    eval_perl = "./conlleval_rev.pl"
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics


def stat(label_predict, label_path, metric_path):
    
    #eval_perl = "./conlleval_rev.pl"
    # write test results
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)

    per = count(label_path, 'PER')
    org = count(label_path, 'ORG')
    loc = count(label_path, 'LOC')
    print(loc[0:3])
    print(org[0:3])
    print(per[0:3])
    print('LOC: recall %.2f prec %.2f F %.2f' % (loc[4]*100, loc[5]*100, loc[6]*100))
    print('ORG: recall %.2f prec %.2f F %.2f' % (org[4]*100, org[5]*100, org[6]*100))
    print('PER: recall %.2f prec %.2f F %.2f' % (per[4]*100, per[5]*100, per[6]*100))

#    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics


def count(label_path, type_entity):
    entity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # T-T, T-F, F-T, F-F, recall, precsion, F-value
    for line in open(label_path, 'r'):
        if line == '\n':continue
        item = line.strip().split(' ')
#        print(item)
        if item[1] == '0':
            if item[2] == '0':
                entity[3] += 1
            elif item[2].split('-')[1] == type_entity:
                entity[2] += 1

        elif item[1].split('-')[1] == type_entity:
            if item[1] == item[2]:
                entity[0] += 1
            else: #item[2] == '0':
                entity[1] += 1

    entity[4] = entity[0]/(entity[0]+entity[1]+0.00001)
    entity[5] = entity[0]/(entity[0]+entity[2]+0.00001)
    entity[6] = entity[4]*entity[5]*2/(entity[4]+entity[5]+0.00001)
    return entity

def static(label_path):

    return results

def count_phrases(label_path, _type):
    sents = []
    sent = []
    label = []
    for line in label_path:
        item = line.strip().split(' ')
        sent.append(item[0])

    return results
