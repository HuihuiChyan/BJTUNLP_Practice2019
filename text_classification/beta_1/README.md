# Introduction
- This code implements kim's [Convolutional Neural Networks for Sentence Classiﬁcation](https://arxiv.org/abs/1408.5882) 
simply
- Use tf.keras API to model in TextCNN.py
## Requirements
- Python 3
- tensorflow 1.12
- Numpy
- Pandas
- gensim

## 1. Description
|type|module|function|notes|
|---|:---|:---|:---|
|-|data_helper.py|utils and data processing|
|-|TextCNN.py|model|
|-|train.py|start of whole project|
|-|get_vocab.py|process Google w2v file|
|d|dataset|datasets

## 2. Training
` python train.py`

## 3. Trick
- half the learning rate every 30 epochs
- 