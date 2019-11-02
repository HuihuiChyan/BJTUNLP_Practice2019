'''
copied from yms
'''

import os
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


def Evaluation(y_true, y_pred):

    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
        digits=4
    )


def conlleval(label_predict):
    """
    :param label_predict:
    :return:
    """
    true = []
    pred = []
    for sent_result in label_predict:
        true_ = []
        pred_ = []
        for char, tag, tag_ in sent_result:
            true_.append(tag)
            pred_.append(tag_)
        true.append(true_)
        pred.append(pred_)
    res = Evaluation(true,pred)
    return res

