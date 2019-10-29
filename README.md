新加入本实验室的同学，请按要求完成下面练习，并提交结果。

### 任务排名(不断更新中)

文本情感分类：

| 排名 | 模型名称          | 准确率 | 提交时间      |
| ---- | ----------------- | ------ | ------------- |
| 1    | beta_1            | 90.60% | 2019年9月18日 |
| 2    | kawhi849CNNs       | 90.40% | 2019年10月26日|
| 3    | tuzikou_1.1      | 89.68% | 2019年10月12日 |
| 4    | gudakoCNNs      | 88.54% | 2019年10月27日 |
| 5    | plz               | 86.58% | 2019年10月19日|
| 6    | AwesomeClassifier | 83.19% | 2019年9月17日 |

命名实体识别：

| 排名 | 模型名称          | F1值 | 提交时间      |
| ---- | ----------------- | ------ | ------------- |
| 1    | kawhi849NER       | 82.30% | 2019年10月13日|
| 2    | zxb-ner           | 78.27% | 2019年10月26日 |

中文分词：

| 排名 | 模型名称          | F1值 | 提交时间      |
| ---- | ----------------- | ------ | ------------- |
| 1    | BMES              | 96.81% | 2019年10月25日|
| 2    | AwesomeSegmentator   | 93.60% | 2019年10月29日|

### 任务内容

| 任务内容                                 | 数据集                | 评价指标 |
| ---------------------------------------- | --------------------- | -------- |
| 1.文本分类(text classification)          | IMDB review           | Accuracy |
| 2.中文分词(chinese word segmentation)    | MSR(from Bakeoff2005) | F1值     |
| 3.命名实体识别(named entity recognition) | CoNLL03               | F1值     |
| 4.文本匹配(text matching)                | SNLI1.0               | F1值     |
| 5.问答系统(Q-A system)                   | WikiQA                | F1值     |
| 6.机器翻译(machine translation)          | UNParallel            | BLEU5    |

### 任务说明

1. 任务难度：任务的设计参考了FudanNLP/nlp-beginner。前四个任务比较简单，也比较基础，第五个和第六个（尤其是第六个）任务比较困难。要求新加入的同学至少选择两个来做，多了不限；
2. 数据集：数据集我已经放在了/home1/huanghui/dataset4practice目录下，划分好了训练集、验证集和测试集；
3. 提交内容：对于测试集的预测结果(精确到小数点后两位)+完整代码+简短的架构描述（命名为README.md），打包发送到18112023@bjtu.edu.cn。可以多次提交，即可以不断更新自己的模型来跑出更高的分数；
4. 其他要求：推荐用深度学习方法，tensorflow或者pytorch二选一（但如果你想用keras也可以）。另外，建议自己设计架构并实现，或者自己复现经典论文。如果直接找写好的开源代码跑一跑，虽然可以马上把分数刷爆，但是意义不大。

### 参考资料

文本分类：

- 可以参照下面两篇论文，分别是TextCNN和TextRNN：
  - https://arxiv.org/abs/1408.5882 : Convolutional Neural Networks for Sentence Classification
  - <http://aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552>：Recurrent convolutional neural networks for text classification
- 或者用更加简单的网络结构也可以实现文本分类的任务；
- 训练集25000句，测试集25000句，需要自己写脚本合在一起；

中文分词：

- <https://www.aclweb.org/anthology/D15-1141>：Long Short-Term Memory Neural Networks for Chinese Word Segmentation 
- Bakeoff2005提供的分词数据有三组，分别是pku,msr和cityu，为了便于比较，请仅使用msr的数据作为训练集和测试集；
- 训练集里有一个空行，预处理的时候请小心；

命名实体识别：

- 可以参照下面这篇论文，是最经典也是最简单的架构：
  - https://arxiv.org/pdf/1603.01360.pdf : Neural Architectures for Named Entity Recognition
- 训练集、验证集(testa)、测试集(testb)都已经分隔好了。禁止将验证集中的数据也用于训练；

文本匹配：

- https://arxiv.org/pdf/1509.06664v1.pdf ：Reasoning about Entailment with Neural Attention
- https://arxiv.org/pdf/1609.06038v3.pdf ：Enhanced LSTM for Natural Language Inference

检索式问答：

- <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf>：Learning Deep Structured Semantic Models for Web Search using Clickthrough Data 
- <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2014_cdssm_final.pdf>：A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval 
- <https://arxiv.org/pdf/1412.6629.pdf>：Semantic Modelling with Long-short-term Memory For Information Retrieval

机器翻译：

- https://arxiv.org/pdf/1706.03762.pdf : Attention Is All You Need
