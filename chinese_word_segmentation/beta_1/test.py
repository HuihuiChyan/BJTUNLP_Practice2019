import tensorflow as tf
from model import CWSModel
from utils import tokenize
from tensorflow.keras import preprocessing
import tensorflow_addons as tfa
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

train_path = '../data/train.utf8'
label_path = '../data/label.utf8'
checkpoint_path = "checkpoints/"

BATCH_SIZE = 128
NUM_HIDDEN = 512
EPOCH = 10
LR = 1e-3
embed_table = None
embed_feature = 300

# 获取text_tokenzier 和 label_tokenizer
text_sequences, text_tokenizer = tokenize(train_path)
label_sequences, label_tokenizer = tokenize(label_path, is_label=True)
vocab_size = len(text_tokenizer.word_index)
num_label = len(label_tokenizer.word_index)

model = CWSModel(NUM_HIDDEN, vocab_size, num_label, embed_table, embed_feature)
# 恢复模型
ckpt = tf.train.Checkpoint(model=model)
ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))

with open('../data/test.utf8', 'r', encoding='utf-8') as fr:
    content = fr.readlines()
content = list(map(lambda x: x.strip(), content))
# print(content[:3])
test_sequences = text_tokenizer.texts_to_sequences(content)
# print(test_sequences[:3])
test_sequences = preprocessing.sequence.pad_sequences(test_sequences, padding='post')
# print(test_sequences[:3])

test_dataset = tf.data.Dataset.from_tensor_slices(test_sequences).batch(BATCH_SIZE)
labels = []
# transition_params = np.load('checkpoints/trans_params.npy')
# transition_params = tf.convert_to_tensor(transition_params)
for batched_test_sequences in test_dataset:
    logits, seq_lens = model(batched_test_sequences)
    for logit, seq_len in zip(logits, seq_lens):
        viterbi_seq, _ = tfa.text.viterbi_decode(logit[:seq_len], model.transition_params)
        labels.append(viterbi_seq)

# print(len(labels))
np.save('../data/pred_labels.npy', labels, allow_pickle=True)





    

