新加入本研究组的同学，请按要求完成下面练习，并提交结果。

### 任务排名(不断更新中)

文本情感分类：

| 排名 | 模型名称          | 准确率 | 提交时间      |
| ---- | ----------------- | ------ | ------------- |
| 1    | kawhi849CNNs       | 93.48% | 2019年11月20日|
| 2    | beta_1            | 90.60% | 2019年9月18日 |
| 3    | tuzikou_1.1      | 89.68% | 2019年10月12日 |
| 4    | venky           | 88.59% | 2019年11月21日 |
| 5    | gudakoCNNs      | 88.54% | 2019年10月27日 |
| 6    | CNNClassifier   | 87.29% | 2019年11月12日 |
| 7    | simlpeClassifier  | 87.27% | 2019年11月9日 |
| 8    | plz               | 86.58% | 2019年10月19日|
| 9    | AwesomeClassifier | 83.19% | 2019年9月17日 |
| 10    | Starry          | 81.45% | 2019年11月4日 |

命名实体识别：

| 排名 | 模型名称          | F1值 | 提交时间      |
| ---- | ----------------- | ------ | ------------- |
| 1    | kawhi849NER       | 82.30% | 2019年10月13日|
| 2    | zxb-ner           | 78.27% | 2019年10月26日 |
| 3    | AwesomeTagger     | 71.37% | 2019年10月30日 |

中文分词：

| 排名 | 模型名称          | F1值 | 提交时间      |
| ---- | ----------------- | ------ | ------------- |
| 1    | kawhi849CWS         | 97.21% | 2019年11月2日|
| 2    | BMES              | 96.81% | 2019年10月25日|
| 3    | AwesomeSegmentator   | 93.60% | 2019年10月29日|
| 4    | awfulCWS         | 91.21% | 2019年11月13日|

文本匹配：

| 排名 | 模型名称          | 准确率 | 提交时间      |
| ---- | ----------------- | ------ | ------------- |
| 1    | AwesomeMatcher    | 67.86% | 2019年11月17日|
| 2    | kawhi849trash     | 64.95% | 2019年10月30日|

### 任务内容

| 任务内容                                 | 数据集                | 评价指标 |
| ---------------------------------------- | --------------------- | -------- |
| 1.文本分类(text classification)          | IMDB review           | Accuracy |
| 2.中文分词(chinese word segmentation)    | MSR(from Bakeoff2005) | F1值     |
| 3.命名实体识别(named entity recognition) | CoNLL03               | F1值     |
| 4.文本匹配(text matching)                | SNLI1.0               | Accuracy |
| 5.问答系统(Q-A system)                   | WebQuestion           | F1值     |
| 6.机器翻译(machine translation)          | UNParallel            | BLEU5    |

### 任务说明

1. 任务难度：任务的设计参考了FudanNLP/nlp-beginner。前四个任务比较简单，也比较基础，第五个和第六个任务比较困难。希望新加入研究组的同学都能做一做，这也是一个入门的过程；
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
- 请使用conlleval.perl官方评测脚本。注意该脚本要求输入的标签集是BIO，而我给大家的词表里是BIOES，需要转换一下；

文本匹配：

- https://arxiv.org/pdf/1509.06664v1.pdf ：Reasoning about Entailment with Neural Attention
- https://arxiv.org/pdf/1609.06038v3.pdf ：Enhanced LSTM for Natural Language Inference
- 建议借助注意力机制实现。如果参考的是ESIM模型（即上面的第二篇论文），可以只用LSTM，忽略Tree-LSTM；

问答系统：

- 之前的检索式问答被我换成基于知识的问答系统(KBQA)了，原因是检索式问答和文本蕴含很像，想给大家找个难一点的任务。不过也正是因此，在这个任务中大家可以学到更多知识，所以有兴趣的同学做一做吧。

机器翻译：

- https://arxiv.org/pdf/1706.03762.pdf : Attention Is All You Need
- 如果你凭自己复现出了上面这篇论文，说明你的编程能力已经很厉害了。
