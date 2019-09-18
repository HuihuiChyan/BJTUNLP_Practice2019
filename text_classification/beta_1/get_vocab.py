"""
get dataset/imdb.vocab embedding table from dataset/GoogleNews-vectors-negative300.bin
save imdb's embedding table into dataset/imdb.w2v.np
"""
import data_helper
import numpy as np
import tensorflow as tf
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('dataset/GoogleNews-vectors-negative300.bin', binary=True)
with open('dataset/imdb.vocab', 'r', encoding='utf-8') as fr:
    content = [word.strip() for word in fr.readlines()]
embedding_table = []
for word in content:
    try:
        embedding_table.append(model[word])
    except: #word not in GoogleNews-w2v
        embedding_table.append(np.array([0] * 300))

# print(embedding_table[89528])

# embedding_table_str = ''
# for item in embedding_table:
#     embedding_table_str += ' '.join(map(str, item))
#     embedding_table_str += '\n'
#
# with open('dataset/imdb.w2v', 'w', encoding='utf-8') as fw:
#     fw.write(embedding_table_str)
np.save('dataset/imdb.w2v.npy', embedding_table)


















