- 仅使用单层Bi-LSTM, 未使用CRF
- 仅在LSTM单元后添加了dropout
- 未使用任何规范化或正则化技术
- 未使用预训练词向量
- 未使用学习率衰减
- LSTM单元忘记添加激活函数了
- 在单张TitanXP上进行训练，直接把整张卡占满了
- 训练10000步时已经完全收敛
- 预测时使用的是BIOES标注集，在输入conlleval.pl时才转化为BIO标注
- 模型参数如下：

|  hyperparams | value  |
| ------------ | ------------ |
|  maxlen      | 50      |
| batch_size   | 64      |
| embedding_size  |  256 |
| lstm_keep_prob  | 0.7  |
| learning_rate  |  0.001 |
| norm_clip      | 5    |
|  lstm_hidden_size |  256 |
