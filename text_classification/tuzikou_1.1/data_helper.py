import data_utlis as da
from data_utlis import read_files
from data_utlis import get_token_data,get_token_text,data_process,get_stop_words_list
from data_utlis import get_vocab
from data_utlis import pad_st,encode_st
import numpy as np

#win_path
data_path = "D:/study/bjtu/nlp_practice/text_classification/data/aclImdb_v1/"
save_path = ""

#root_path
#data_path = ""
maxlen = 300

train_data = read_files(data_path,"train")
test_data = read_files(data_path,"test")
print("read_file success!")

train_token = get_token_data(train_data)
test_token = get_token_data(test_data)

print("get_token_data success!")

vocab,vocab_size,word_to_idx,idx_to_word = get_vocab(train_token)
np.save("vocab.npy",vocab)

print("vocab_save success!")

train_features = pad_st(encode_st(train_token, vocab,word_to_idx),maxlen)
test_features = pad_st(encode_st(test_token, vocab,word_to_idx),maxlen)
train_label = [score for _, score in train_data]
test_label = [score for _, score in test_data]

print("get_feature_data success!")

np.save("train_features.npy",train_features)
np.save("test_features.npy",test_features)
np.save("train_label.npy",train_label)
np.save("test_label.npy",test_label)


