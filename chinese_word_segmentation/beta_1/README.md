# 汉语分词 TensorFlow2.0

## Introduction
- This Chinese Word Segmentation implements with TensorFlow 2.0

## Requirements
- Python 3
- TensorFlow 2.0.0
- TensorFlow_addons
- Numpy

## Description
|module|function|notes|
|---|:---|:---|
|utils.py| 
|model.py|CWS Model|use tensorflow_addons(crf_log_likelihood)
|train.py|train model | hyperparameters
|test.py |predict attribute of characters| BMES
|utils_eval.py|convert predict labels into predict text| get text_pred.utf8
|eva/icwb2-data/scripts/score| score scripts|get R,P,F1 value

## Reference
1. [My tensorflow 1.12 version CWS](https://github.com/RisanLi/BiLSTM_CNN-CRF-CWS)

## Tips
1. How to use evaluation script have written in [My tensorflow 1.12 version CWS README](https://github.com/RisanLi/BiLSTM_CNN-CRF-CWS)

