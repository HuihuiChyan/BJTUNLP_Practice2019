Introduction
This code is task for NER On CoNLL03 data set
Use tensorflow Framework
Requirements
Python 3
tensorflow 1.12
Numpy
sklearn

1. Description
Use Bi-LSTM and Attention mechanism to model this task,but now the performance is 
not perfect ,I will improve it latter.

2. Training
CUDA_VISIBLE_DEVICES=6 python3 ner_lstm_crf_attn.py --batch_size 8 --epoch 300


3.Result
�������һ�֣�������Weight�������Ľ�������������

Ŀǰ��ý����
F1:78.27

As we can see.
Location message recoginze rate is best ,and org,Misc rate is not very well
I will improve it.