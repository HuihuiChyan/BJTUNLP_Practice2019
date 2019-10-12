| model | ACC | epoch | batch_size | lr |dropout|embedding_size|ps|
|--------|--------|--------|
| cnn_pytorch |     87.04    |  1000 |  64  |  0.005 |0.05|300(random)|filter Windows（3，4，5）|
| cnn_pytorch |     88.60    |  1000 |  64  |  0.05 |0.05|300(random)|filter Windows（3，4，5）|
| cnn_pytorch |     87.13    |  1000 |  64  |  0.05 |0.1|300(random)|filter Windows（3，4，5）|
| cnn_pytorch |     89.04    |  1000 |  64  |  0.01 |0.05|300(random)|filter Windows（3，4，5）|
| cnn_pytorch |     89.68    |  1000 |  64  |  0.01 |0.01|300(random)|filter Windows（3，4，5）|








###备注
- 数据处理完直接保存成向量文件，方便很多

- 文本预处理有点问题，待解决

- 本来想用预训练好的embedding，结果效果不太好，待解决
- 减小dropout值模型效果有提升

- 一开始学习率设置成0.001结果很差，不到80

- 加了学习率衰减的策略，暂时没有对比是否有效果

- （随缘改进和尝试新模型）

- glove与word2vec的转换参考了https://blog.csdn.net/nlpuser/article/details/83627709
