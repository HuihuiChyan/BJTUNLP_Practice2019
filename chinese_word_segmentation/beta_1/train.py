from utils import tokenize
import tensorflow as tf
from tensorflow import keras
from model import CWSModel
import tensorflow_addons as tfa
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

train_path = '../data/train.utf8'
label_path = '../data/label.utf8'

BATCH_SIZE = 256
NUM_HIDDEN = 512
EPOCH = 40
LR = 1e-3

text_sequences, text_tokenizer = tokenize(train_path)
label_sequences, label_tokenizer = tokenize(label_path, is_label=True)

vocab_size = len(text_tokenizer.word_index)
num_label = len(label_tokenizer.word_index)
embed_table = None
embed_feature = 300

# print('text_sequences[:2]')
# print(text_sequences[:2])
# print('text_tokenizer.word_index[:2]', end=' ')
# print(list(text_tokenizer.word_index.items())[:2])
# print('label_tokenzier.word_index', end=' ')
# print(label_tokenizer.word_index)
# print(len(text_sequences))
# print(len(label_sequences))

train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, label_sequences))
train_dataset = train_dataset.shuffle(len(text_sequences)).batch(BATCH_SIZE, drop_remainder=True)




model = CWSModel(NUM_HIDDEN, vocab_size, num_label, embed_table, embed_feature)
optimizer = keras.optimizers.Adam(LR)
checkpoint_path = "checkpoints/"

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, checkpoint_name='model.ckpt',max_to_keep=3)

# print([i for i in model.trainable_variables])

for epoch in range(EPOCH):
    for step, (batch_text_sequences, batch_label_sequences) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits, seq_lens, log_likelihood = model(batch_text_sequences, batch_label_sequences, training=True)
            loss = -tf.reduce_mean(log_likelihood)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if step % 100 == 0:
            print('epoch %d, step %d, loss %.4f' % (epoch, step, loss))
    ckpt_save_path = ckpt_manager.save()
    print(model.transition_params.shape)
    # np.save('checkpoints/trans_params.npy', model.transition_params.numpy())

