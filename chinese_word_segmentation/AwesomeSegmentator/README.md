- 只使用了一层Bi-LSTM，没有使用CRF。

- 添加了梯度截断。

- 没有使用任何正则化和规范化技术。

- 除了LSTM单元，其他地方没有使用dropout。

- 没有使用学习率衰减。

- 训练8000步时就已经基本收敛。

  具体参数如下：

| params          | value |
| --------------- | ----- |
| max_len         | 120   |
| batch_size      | 256   |
| vocab_size      | 4002  |
| learning_rate   | 1e-3  |
| norm_clip       | 5.0   |
| embedding_size  | 200   |
| rnn_hidden_size | 256   |
| rnn_keep_prob   | 0.7   |

