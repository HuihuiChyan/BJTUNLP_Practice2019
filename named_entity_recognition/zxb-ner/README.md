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

tokens with 5842 phrases; found: 5954 phrases; correct: 4201.
Total:precision:  70.56%; recall:  71.91%; FB1:  71.23
LOC:  precision:  78.34%; recall:  83.45%; FB1:  80.82  1796
MISC: precision:  66.49%; recall:  69.37%; FB1:  67.90  746
ORG:  precision:  64.63%; recall:  61.15%; FB1:  62.84  1634
PER:  precision:  69.85%; recall:  72.46%; FB1:  71.13  1778

As we can see.
Location message recoginze rate is best ,and org,Misc rate is not very well
I will improve it.